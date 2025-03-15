
import mongoose from 'mongoose';

// MongoDB connection string - to be filled in by the user
const MONGODB_URI = process.env.MONGODB_URI || '';

// Connection options
const options = {
  useNewUrlParser: true,
  useUnifiedTopology: true,
} as mongoose.ConnectOptions;

let isConnected = false;

export const connectToDatabase = async () => {
  if (isConnected) {
    console.log('Using existing MongoDB connection');
    return;
  }

  if (!MONGODB_URI) {
    console.error('MongoDB URI is not defined. Please set the MONGODB_URI environment variable.');
    throw new Error('MongoDB URI is not defined');
  }

  try {
    await mongoose.connect(MONGODB_URI, options);
    isConnected = true;
    console.log('Connected to MongoDB');
  } catch (error) {
    console.error('Error connecting to MongoDB:', error);
    throw error;
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
