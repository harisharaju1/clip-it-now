import { env } from "~/env";
import { inngest } from "./client";

export const processVideo = inngest.createFunction(
  {
    id: "process-video",
    concurrency: {
      limit: 1,
      key: "event.data.userId",
    },
  },
  { event: "process-video-events" },
  async ({ event, step }) => {
    await step.run("call-modal-endpoint", async () => {
      // Your video processing logic here
      console.log("Processing video with event data:", event.data);
      // Simulate processing delay
      await fetch(env.PROCESS_VIDEO_ENDPOINT, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${env.PROCESS_VIDEO_ENDPOINT_AUTH}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ s3_key: "test1/vod1.mp4" }),
      });
      console.log("Video processed successfully");
    });
  },
);
