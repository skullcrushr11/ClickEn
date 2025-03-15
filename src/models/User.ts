
import mongoose, { Document, Schema } from 'mongoose';

export interface IUser extends Document {
  email: string;
  password: string;
  userType: 'student' | 'organizer';
  firstName?: string;
  lastName?: string;
  createdAt: Date;
  updatedAt: Date;
}

const UserSchema: Schema = new Schema(
  {
    email: { type: String, required: true, unique: true },
    password: { type: String, required: true },
    userType: { type: String, enum: ['student', 'organizer'], required: true },
    firstName: { type: String },
    lastName: { type: String },
  },
  { timestamps: true }
);

// Check if the model already exists to prevent model overwrite errors
export const User = mongoose.models.User || mongoose.model<IUser>('User', UserSchema);
