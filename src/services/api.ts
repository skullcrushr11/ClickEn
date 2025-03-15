
import { connectToDatabase } from '../config/mongodb';
import { User, IUser } from '../models/User';
import { Question, IQuestion } from '../models/Question';
import { TestSession, ITestSession } from '../models/TestSession';
import { UserSubmission, IUserSubmission } from '../models/UserSubmission';

// Authentication Services
export const authenticateUser = async (email: string, password: string) => {
  await connectToDatabase();
  
  // In a real application, you would hash the password before comparing
  // For simplicity, we're doing a direct comparison here
  const user = await User.findOne({ email, password }).exec();
  
  if (!user) {
    throw new Error('Invalid credentials');
  }
  
  return {
    id: user._id,
    email: user.email,
    userType: user.userType,
    firstName: user.firstName,
    lastName: user.lastName
  };
};

export const registerUser = async (userData: Partial<IUser>) => {
  await connectToDatabase();
  
  // Check if user already exists
  const existingUser = await User.findOne({ email: userData.email }).exec();
  
  if (existingUser) {
    throw new Error('User already exists');
  }
  
  // Create new user
  const newUser = new User(userData);
  await newUser.save();
  
  return {
    id: newUser._id,
    email: newUser.email,
    userType: newUser.userType
  };
};

// Question Services
export const getQuestions = async () => {
  await connectToDatabase();
  return await Question.find().exec();
};

export const getQuestionById = async (id: string) => {
  await connectToDatabase();
  return await Question.findById(id).exec();
};

export const createQuestion = async (questionData: Partial<IQuestion>) => {
  await connectToDatabase();
  const newQuestion = new Question(questionData);
  await newQuestion.save();
  return newQuestion;
};

// Test Session Services
export const getTestSessions = async (userId: string, userType: string) => {
  await connectToDatabase();
  
  if (userType === 'organizer') {
    return await TestSession.find({ createdBy: userId }).exec();
  } else {
    // For students, find test sessions they are candidates for
    return await TestSession.find({ candidates: userId, status: 'active' }).exec();
  }
};

export const getTestSessionById = async (id: string) => {
  await connectToDatabase();
  return await TestSession.findById(id).populate('questions').exec();
};

export const createTestSession = async (sessionData: Partial<ITestSession>) => {
  await connectToDatabase();
  const newSession = new TestSession(sessionData);
  await newSession.save();
  return newSession;
};

// User Submission Services
export const getUserSubmissions = async (userId: string, testSessionId: string) => {
  await connectToDatabase();
  return await UserSubmission.find({ 
    user: userId,
    testSession: testSessionId 
  }).populate('question').exec();
};

export const saveUserSubmission = async (submissionData: Partial<IUserSubmission>) => {
  await connectToDatabase();
  
  // Check if a submission already exists
  const existingSubmission = await UserSubmission.findOne({
    user: submissionData.user,
    question: submissionData.question,
    testSession: submissionData.testSession
  }).exec();
  
  if (existingSubmission) {
    // Update existing submission
    Object.assign(existingSubmission, submissionData);
    await existingSubmission.save();
    return existingSubmission;
  } else {
    // Create new submission
    const newSubmission = new UserSubmission(submissionData);
    await newSubmission.save();
    return newSubmission;
  }
};

// Analytics for Organizer Dashboard
export const getOrganizerInsights = async (organizerId: string) => {
  await connectToDatabase();
  
  const testSessions = await TestSession.find({ createdBy: organizerId }).exec();
  
  const sessionIds = testSessions.map(session => session._id);
  
  const submissions = await UserSubmission.find({
    testSession: { $in: sessionIds }
  }).exec();
  
  // Calculate stats
  const totalCandidates = await User.countDocuments({ 
    _id: { $in: testSessions.flatMap(session => session.candidates) }
  }).exec();
  
  const completedCandidates = await UserSubmission.countDocuments({
    testSession: { $in: sessionIds },
    completed: true
  }).exec();
  
  const flaggedCandidates = await UserSubmission.countDocuments({
    testSession: { $in: sessionIds },
    flaggedForCheating: true
  }).exec();
  
  return {
    totalSessions: testSessions.length,
    totalCandidates,
    completedCandidates,
    flaggedCandidates,
    testSessions: testSessions.map(session => ({
      id: session._id,
      title: session.title,
      status: session.status,
      startDate: session.startDate,
      endDate: session.endDate,
      totalCandidates: session.candidates.length,
      completedCandidates: submissions.filter(
        sub => sub.testSession.toString() === session._id.toString() && sub.completed
      ).length,
      flaggedCandidates: submissions.filter(
        sub => sub.testSession.toString() === session._id.toString() && sub.flaggedForCheating
      ).length
    }))
  };
};
