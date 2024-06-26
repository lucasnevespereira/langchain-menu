import { PromptTemplate } from "@langchain/core/prompts";
import { OpenAI } from "@langchain/openai";
import { LLMChain, SequentialChain } from "langchain/chains";
import { StructuredOutputParser } from "langchain/output_parsers";
import { z } from "zod";
import fs from "fs";
import dotenv from "dotenv";
dotenv.config();

const openAIkey = process.env.OPENAI_API_KEY || "YOUR_OPEN_API_KEY";

const llm = new OpenAI({
  model: "gpt-3.5-turbo",
  maxTokens: 4096,
  maxRetries: 5,
  configuration: {
    apiKey: openAIkey,
  },
});

const foodItem = z.string().describe("food item");
const quantity = z.string().describe("quantity in grams");
const foodItemWithQuantity = z
  .object({ food: foodItem, quantity: quantity })
  .describe("food item with quantity");
const menuContentParser = StructuredOutputParser.fromZodSchema(
  z
    .object({
      breakfast: z.array(foodItemWithQuantity).describe("Breakfast"),
      morning_snack: z.array(foodItemWithQuantity).describe("Morning snack"),
      lunch: z.array(foodItemWithQuantity).describe("Lunch"),
      afternoon_snack: z
        .array(foodItemWithQuantity)
        .describe("Afternoon snack"),
      dinner: z.array(foodItemWithQuantity).describe("Dinner"),
      evening_snack: z.array(foodItemWithQuantity).describe("Evening snack"),
    })
    .describe("Menu")
);
const groceryListContentParser = StructuredOutputParser.fromZodSchema(
  z.array(foodItemWithQuantity).describe("Grocery list")
);
const overallParser = StructuredOutputParser.fromZodSchema(
  z.object({
    menu: menuContentParser,
    grocery_list: groceryListContentParser,
  })
);

// Chain to generate the daily menu
const dailyMenuTemplate = new PromptTemplate({
  template:
    "You are a nutritionist with 20 years of experience. Please generate a daily menu in {language} for a person the following profile: ${profile}. Respect the following format: ${format_instructions}. Additional notes: {additionalNotes}",
  inputVariables: ["language", "profile", "additionalNotes"],
  partialVariables: {
    format_instructions: menuContentParser.getFormatInstructions(),
  },
  outputParser: menuContentParser,
});

const dailyMenuChain = new LLMChain({
  llm,
  prompt: dailyMenuTemplate,
  outputKey: "menu",
});

// Chain to generate the grocery list based on the menu
const groceryListTemplate = new PromptTemplate({
  template:
    "As a nutrionist for the following menu: {menu}, please create a grocery list for your client in {language}. You should respect the following format: ${format_instructions}.",
  inputVariables: ["language", "menu"],
  partialVariables: {
    format_instructions: groceryListContentParser.getFormatInstructions(),
  },
  outputParser: groceryListContentParser,
});
const groceryListChain = new LLMChain({
  llm,
  prompt: groceryListTemplate,
  outputKey: "grocery_list",
});

// Combine the chains into a sequential chain
const overallChain = new SequentialChain({
  chains: [dailyMenuChain, groceryListChain],
  inputVariables: ["language", "profile", "additionalNotes"],
  outputVariables: ["menu", "grocery_list"],
  outputParser: overallParser,
});

// Example usage
const profile = {
  age: 30,
  weight: 70,
  height: 175,
  gender: "male",
  activityLevel: "moderate",
  weightGoal: 68,
  weightLossPerWeek: 0.5,
  dailyCalories: 2000,
  allergies: ["milk"],
  regimes: [],
  measurementSystem: "METRIC",
};

const result = await overallChain.call({
  language: "English",
  profile: profile,
  additionalNotes: "No sugar",
});

console.log(result);
fs.writeFileSync("result.json", JSON.stringify(result, null, 2));
