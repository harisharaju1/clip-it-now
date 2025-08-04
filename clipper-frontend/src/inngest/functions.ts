import { env } from "~/env";
import { inngest } from "./client";
import { db } from "~/server/db";
import { ListObjectsV2Command, S3Client } from "@aws-sdk/client-s3";

export const processVideo = inngest.createFunction(
  {
    id: "process-video",
    retries: 1,
    name: "Process Video",
    concurrency: {
      limit: 1,
      key: "event.data.userId",
    },
  },
  {
    event: "process-video-events",
  },
  async ({ event, step }) => {
    const { uploadedFileId } = event.data as {
      uploadedFileId: string;
      userId: string;
    };

    try {
      const { userId, credits, s3Key } = await step.run(
        "check-credits",
        async () => {
          // unsafe-call, unsafe-member-access
          const uploadedFile = await db.uploadedFile.findUniqueOrThrow({
            where: { id: uploadedFileId },
            select: {
              user: {
                select: {
                  id: true,
                  credits: true,
                },
              },
              s3Key: true,
            },
          });

          return {
            userId: uploadedFile.user.id,
            credits: uploadedFile.user.credits,
            s3Key: uploadedFile.s3Key,
          };
        },
      );

      if (credits > 0) {
        // Set the status to processing
        await step.run("set-status-processing", async () => {
          await db.uploadedFile.update({
            where: {
              id: uploadedFileId,
            },
            data: {
              status: "processing",
            },
          });
        });
        // Call the modal endpoint to process the video
        await step.run("call-modal-endpoint", async () => {
          console.log("Processing video with event data:", event.data);
          await fetch(env.PROCESS_VIDEO_ENDPOINT, {
            method: "POST",
            headers: {
              Authorization: `Bearer ${env.PROCESS_VIDEO_ENDPOINT_AUTH}`,
              "Content-Type": "application/json",
            },
            body: JSON.stringify({ s3_key: s3Key }),
          });
          console.log("Video processed successfully");
        });
        // Create clips in DB
        const clipsFound = await step.run("create-clips-in-db", async () => {
          const folderPrefix = s3Key.split("/")[0];

          const allKeys = await listS3ObjectsByPrefix(folderPrefix);

          const clipKeys = allKeys.filter(
            (key): key is string =>
              key !== undefined && !key.endsWith("original.mp4"),
          );

          if (clipKeys.length === 0) {
            throw new Error("No clips found in S3 for the given prefix");
          }

          await db.clip.createMany({
            data: clipKeys.map((clipKey) => ({
              s3Key: clipKey,
              uploadedFileId,
              userId,
            })),
          });

          return clipKeys.length;
        });
        // Deduct credits for User
        await step.run("deduct-credits", async () => {
          await db.user.update({
            where: { id: userId },
            data: {
              credits: {
                decrement: Math.min(clipsFound, credits),
              },
            },
          });
        });
        // Set the status to processed
        await step.run("set-status-processed", async () => {
          await db.uploadedFile.update({
            where: {
              id: uploadedFileId,
            },
            data: {
              status: "processed",
            },
          });
        });
      } else {
        // Set the status to failed if no credits
        await step.run("set-status-no-credits", async () => {
          await db.uploadedFile.update({
            where: {
              id: uploadedFileId,
            },
            data: {
              status: "no credits",
            },
          });
        });
      }
    } catch (error: unknown) {
      await db.uploadedFile.update({
        where: {
          id: uploadedFileId,
        },
        data: {
          status:
            "failed" + (error instanceof Error ? `: ${error.message}` : ""),
        },
      });
    }
  },
);

async function listS3ObjectsByPrefix(prefix: string) {
  const s3Client = new S3Client({
    region: env.AWS_REGION,
    credentials: {
      accessKeyId: env.AWS_ACCESS_KEY_ID,
      secretAccessKey: env.AWS_SECRET_ACCESS_KEY,
    },
  });

  const listCommand = new ListObjectsV2Command({
    Bucket: env.S3_BUCKET_NAME,
    Prefix: prefix,
  });

  const response = await s3Client.send(listCommand);
  if (!response.Contents) {
    return [];
  }
  return response.Contents?.map((item) => item.Key).filter(Boolean) ?? [];
}
