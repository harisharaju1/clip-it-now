"use server";

import Stripe from "stripe";
import { hashPassword } from "~/lib/auth";
import { signupSchema, type SignupFormValues } from "~/schemas/auth";
import { db } from "~/server/db";

type SignupResult = {
  success: boolean;
  error?: string;
};

export async function signUp(data: SignupFormValues): Promise<SignupResult> {
  // Validate the input data against the schema
  const validationResult = signupSchema.safeParse(data);
  if (!validationResult.success) {
    return {
      success: false,
      error: validationResult.error.issues[0]?.message || "Invalid input",
    };
  }

  const { email, password } = validationResult.data;
  try {
    // Check if the user already exists
    const existingUser = await db.user.findUnique({
      where: { email },
    });

    if (existingUser) {
      return {
        success: false,
        error: "User already exists",
      };
    }

    // Hash the password before saving
    const hashedPassword = await hashPassword(password);

    // Create a new Stripe customer
    // const stripe = new Stripe(process.env.STRIPE_SECRET_KEY);
    // const stripeCustomer = await stripe.customers.create({
    //   email: email.toLowerCase(),
    // });

    // Create the user in the database
    await db.user.create({
      data: {
        email: email.toLowerCase(),
        password: hashedPassword,
        // stripeCustomerId: stripeCustomer.id, // Store Stripe customer ID
      },
    });

    return {
      success: true,
    };
  } catch (error) {
    console.error("Error during signup:", error);
    throw error;
  }
}
