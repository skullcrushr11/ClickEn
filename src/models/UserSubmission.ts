
import mongoose, { Document, Schema } from 'mongoose';

export interface IUserSubmission extends Document {
  testSession: mongoose.Types.ObjectId;
  question: mongoose.Types.ObjectId;
  user: mongoose.Types.ObjectId;
  code: string;
  language: string;
  status: 'started' | 'in-progress' | 'completed' | 'timed-out';
  result: 'passed' | 'failed' | 'pending' | null;
  timeSpent: number;
  completed: boolean;
  flaggedForCheating: boolean;
  flaggedReasons?: string[];
  createdAt: Date;
  updatedAt: Date;
}

const UserSubmissionSchema: Schema = new Schema(
  {
    testSession: { type: Schema.Types.ObjectId, ref: 'TestSession', required: true },
    question: { type: Schema.Types.ObjectId, ref: 'Question', required: true },
    user: { type: Schema.Types.ObjectId, ref: 'User', required: true },
    code: { type: String },
    language: { type: String },
    status: { 
      type: String, 
      enum: ['started', 'in-progress', 'completed', 'timed-out'], 
      default: 'started' 
    },
    result: { 
      type: String, 
      enum: ['passed', 'failed', 'pending', null], 
      default: null 
    },
    timeSpent: { type: Number, default: 0 },
    completed: { type: Boolean, default: false },
    flaggedForCheating: { type: Boolean, default: false },
    flaggedReasons: [{ type: String }],
  },
  { timestamps: true }
);

export const UserSubmission = mongoose.models.UserSubmission || 
  mongoose.model<IUserSubmission>('UserSubmission', UserSubmissionSchema);
