import mongoose, { Document, Schema } from 'mongoose';

interface Example {
  input: string;
  output: string;
  explanation?: string;
}

interface Option {
  text: string;
  isCorrect: boolean;
}

export interface IQuestion extends Document {
  title: string;
  description: string;
  difficulty: 'easy' | 'medium' | 'hard';
  type: 'coding' | 'mcq' | 'subjective';
  timeLimit: number;
  constraints: string[];
  tags: string[];
  timeComplexity?: string;
  spaceComplexity?: string;
  options?: Option[];  // Added for MCQ questions
  starterCodeJs: string;
  starterCodePython: string;
  starterCodeJava: string;
  starterCodeCpp: string;
  starterCodeGolang: string;
  examples: Example[];
  createdBy: mongoose.Types.ObjectId;
  createdAt: Date;
  updatedAt: Date;
}

const QuestionSchema: Schema = new Schema(
  {
    title: { type: String, required: true },
    description: { type: String, required: true },
    difficulty: { type: String, enum: ['easy', 'medium', 'hard'], required: true },
    type: { type: String, enum: ['coding', 'mcq', 'subjective'], required: true },
    timeLimit: { type: Number, required: true },
    constraints: [{ type: String }],
    tags: [{ type: String }],
    timeComplexity: { type: String },
    spaceComplexity: { type: String },
    options: [
      {
        text: { type: String },
        isCorrect: { type: Boolean },
      },
    ],
    starterCodeJs: { type: String, required: true },
    starterCodePython: { type: String, required: true },
    starterCodeJava: { type: String, required: true },
    starterCodeCpp: { type: String, required: true },
    starterCodeGolang: { type: String, required: true },
    examples: [
      {
        input: { type: String, required: true },
        output: { type: String, required: true },
        explanation: { type: String },
      },
    ],
    createdBy: { type: Schema.Types.ObjectId, ref: 'User', required: true },
  },
  { timestamps: true }
);

// Check if we're in a Node.js environment where mongoose is fully available
let Question;
if (typeof window === 'undefined' && mongoose.connection?.readyState) {
  Question = mongoose.models.Question || mongoose.model<IQuestion>('Question', QuestionSchema);
} else {
  // We're in a browser environment, so we'll use a mock or null
  Question = null;
}

export { Question };