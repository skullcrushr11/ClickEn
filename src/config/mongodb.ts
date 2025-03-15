
import mongoose, { Types } from 'mongoose';

// MongoDB connection string - to be filled in by the user
const MONGODB_URI = import.meta.env.VITE_MONGODB_URI || '';

// Connection options
const options = {
  useNewUrlParser: true,
  useUnifiedTopology: true,
} as mongoose.ConnectOptions;

let isConnected = false;

export const connectToDatabase = async (): Promise<boolean> => {
  if (isConnected) {
    console.log('Using existing MongoDB connection');
    return true;
  }

  if (!MONGODB_URI) {
    console.log('MongoDB URI is not defined. Using mock data instead.');
    return false;
  }

  try {
    await mongoose.connect(MONGODB_URI, options);
    isConnected = true;
    console.log('Connected to MongoDB');
    return true;
  } catch (error) {
    console.error('Error connecting to MongoDB:', error);
    console.log('Using mock data instead');
    return false;
  }
};

export const disconnectFromDatabase = async () => {
  if (!isConnected) {
    return;
  }

  try {
    await mongoose.disconnect();
    isConnected = false;
    console.log('Disconnected from MongoDB');
  } catch (error) {
    console.error('Error disconnecting from MongoDB:', error);
    throw error;
  }
};

export const getMongoConnection = () => {
  return mongoose.connection;
};

export const isMongoConnected = () => {
  return isConnected;
};

// Helper function to convert string IDs to ObjectIds
export const toObjectId = (id: string): Types.ObjectId => {
  return new Types.ObjectId(id);
};

// Helper to stringify ObjectIds
export const toStringId = (id: Types.ObjectId | string): string => {
  return id.toString();
};
