
import mongoose, { Document, Schema } from 'mongoose';

export interface ITestSession extends Document {
  title: string;
  description?: string;
  duration: number;
  startDate: Date;
  endDate: Date;
  enableProctoring: boolean;
  monitorKeystrokes?: boolean;
  monitorMouseMovements?: boolean;
  preventTabSwitching?: boolean;
  preventCopyPaste?: boolean;
  questions: mongoose.Types.ObjectId[];
  candidates: mongoose.Types.ObjectId[];
  createdBy: mongoose.Types.ObjectId;
  status: 'draft' | 'active' | 'completed';
  createdAt: Date;
  updatedAt: Date;
}

const TestSessionSchema: Schema = new Schema(
  {
    title: { type: String, required: true },
    description: { type: String },
    duration: { type: Number, required: true },
    startDate: { type: Date, required: true },
    endDate: { type: Date, required: true },
    enableProctoring: { type: Boolean, default: true },
    monitorKeystrokes: { type: Boolean, default: true },
    monitorMouseMovements: { type: Boolean, default: true },
    preventTabSwitching: { type: Boolean, default: true },
    preventCopyPaste: { type: Boolean, default: true },
    questions: [{ type: Schema.Types.ObjectId, ref: 'Question' }],
    candidates: [{ type: Schema.Types.ObjectId, ref: 'User' }],
    createdBy: { type: Schema.Types.ObjectId, ref: 'User', required: true },
    status: { type: String, enum: ['draft', 'active', 'completed'], default: 'draft' },
  },
  { timestamps: true }
);

export const TestSession = mongoose.models.TestSession || mongoose.model<ITestSession>('TestSession', TestSessionSchema);
