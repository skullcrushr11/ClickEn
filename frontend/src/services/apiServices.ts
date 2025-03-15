// src/services/apiService.ts

import { mockData } from './mockData';

// Types
export interface IUser {
  _id: string;
  email: string;
  password: string;
  userType: 'student' | 'organizer';
  firstName?: string;
  lastName?: string;
  createdAt: Date;
  updatedAt: Date;
}

export interface IQuestion {
  _id: string;
  title: string;
  description: string;
  difficulty: 'easy' | 'medium' | 'hard';
  timeLimit: number;
  constraints: string[];
  tags: string[];
  starterCodeJs: string;
  starterCodePython: string;
  starterCodeJava: string;
  starterCodeCpp: string;
  starterCodeGolang: string;
  examples: {
    input: string;
    output: string;
    explanation?: string;
  }[];
  createdBy: string;
  createdAt: Date;
  updatedAt: Date;
}

export interface ITestSession {
  _id: string;
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
  questions: string[] | IQuestion[];
  candidates: string[];
  createdBy: string;
  status: 'draft' | 'active' | 'completed';
  createdAt: Date;
  updatedAt: Date;
}

export interface IUserSubmission {
  _id: string;
  testSession: string;
  question: string | IQuestion;
  user: string;
  code: string;
  language: string;
  status: string;
  result: string;
  timeSpent: number;
  completed: boolean;
  flaggedForCheating: boolean;
  createdAt: Date;
  updatedAt: Date;
}

// Authentication Services
export const authenticateUser = async (email: string, password: string) => {
  // In a real app, this would make an API call
  const user = mockData.users.find(u => u.email === email && u.password === password);
    
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
  // In a real app, this would make an API call
  const existingUser = mockData.users.find(u => u.email === userData.email);
    
  if (existingUser) {
    throw new Error('User already exists');
  }
  
  const newUser = {
    _id: `user${mockData.users.length + 1}`,
    ...userData,
    createdAt: new Date(),
    updatedAt: new Date()
  };
  
  mockData.users.push(newUser as any);
  
  return {
    id: newUser._id,
    email: userData.email,
    userType: userData.userType
  };
};

// Question Services
export const getQuestions = async () => {
  // In a real app, this would make an API call
  return mockData.questions;
};

export const getQuestionById = async (id: string) => {
  // In a real app, this would make an API call
  return mockData.questions.find(q => q._id === id) || null;
};

// Test Session Services
export const getTestSessions = async (userId: string, userType: string) => {
  // In a real app, this would make an API call
  if (userType === 'organizer') {
    return mockData.testSessions.filter(session => session.createdBy === userId);
  } else {
    return mockData.testSessions.filter(
      session => session.candidates.includes(userId) && session.status === 'active'
    );
  }
};

export const getTestSessionById = async (id: string) => {
  // In a real app, this would make an API call
  const session = mockData.testSessions.find(s => s._id === id);
  if (session) {
    // Simulate populated questions
    return {
      ...session,
      questions: session.questions.map(qId => 
        mockData.questions.find(q => q._id === qId)
      )
    };
  }
  return null;
};

// User Submission Services
export const getUserSubmissions = async (userId: string, testSessionId: string) => {
  // In a real app, this would make an API call
  const submissions = mockData.userSubmissions.filter(
    sub => sub.user === userId && sub.testSession === testSessionId
  );
  
  // Simulate populated questions
  return submissions.map(sub => ({
    ...sub,
    question: mockData.questions.find(q => q._id === sub.question)
  }));
};

export const saveUserSubmission = async (submissionData: Partial<IUserSubmission>) => {
  // In a real app, this would make an API call
  const userStr = submissionData.user?.toString();
  const questionStr = submissionData.question?.toString();
  const testSessionStr = submissionData.testSession?.toString();
  
  // Check if a submission already exists
  const existingIndex = mockData.userSubmissions.findIndex(
    sub => 
      sub.user === userStr && 
      sub.question === questionStr && 
      sub.testSession === testSessionStr
  );
  
  if (existingIndex !== -1) {
    // Update existing submission
    mockData.userSubmissions[existingIndex] = {
      ...mockData.userSubmissions[existingIndex],
      ...submissionData,
      user: userStr as string,
      question: questionStr as string,
      testSession: testSessionStr as string,
      updatedAt: new Date()
    };
    return mockData.userSubmissions[existingIndex];
  } else {
    // Create new submission
    const newSubmission = {
      _id: `sub${mockData.userSubmissions.length + 1}`,
      ...submissionData,
      user: userStr as string,
      question: questionStr as string,
      testSession: testSessionStr as string,
      createdAt: new Date(),
      updatedAt: new Date()
    };
    
    mockData.userSubmissions.push(newSubmission as any);
    return newSubmission;
  }
};

// Analytics for Organizer Dashboard
export const getOrganizerInsights = async (organizerId: string) => {
  // In a real app, this would make an API call
  const testSessions = mockData.testSessions.filter(session => session.createdBy === organizerId);
  const sessionIds = testSessions.map(session => session._id);
  
  const submissions = mockData.userSubmissions.filter(
    sub => sessionIds.includes(sub.testSession as string)
  );
  
  // Calculate stats
  const candidateIds = new Set(testSessions.flatMap(session => session.candidates));
  const totalCandidates = candidateIds.size;
  
  const completedCandidates = new Set(
    submissions.filter(sub => sub.completed).map(sub => sub.user)
  ).size;
  
  const flaggedCandidates = new Set(
    submissions.filter(sub => sub.flaggedForCheating).map(sub => sub.user)
  ).size;
  
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
        sub => sub.testSession === session._id && sub.completed
      ).length,
      flaggedCandidates: submissions.filter(
        sub => sub.testSession === session._id && sub.flaggedForCheating
      ).length
    }))
  };
};