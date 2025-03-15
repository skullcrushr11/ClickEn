
import mongoose, { Document, Schema } from 'mongoose';

interface Example {
  input: string;
  output: string;
  explanation?: string;
}

export interface IQuestion extends Document {
  title: string;
  description: string;
  difficulty: 'easy' | 'medium' | 'hard';
  timeLimit: number;
  constraints: string[];
  tags: string[];
  timeComplexity?: string;
  spaceComplexity?: string;
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
    timeLimit: { type: Number, required: true },
    constraints: [{ type: String }],
    tags: [{ type: String }],
    timeComplexity: { type: String },
    spaceComplexity: { type: String },
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

export const Question = mongoose.models.Question || mongoose.model<IQuestion>('Question', QuestionSchema);
