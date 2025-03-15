
import { connectToDatabase, toObjectId, toStringId } from '../config/mongodb';
import { User, IUser } from '../models/User';
import { Question, IQuestion } from '../models/Question';
import { TestSession, ITestSession } from '../models/TestSession';
import { UserSubmission, IUserSubmission } from '../models/UserSubmission';

// Mock data for when MongoDB is not connected
const mockData = {
  users: [
    {
      _id: 'user1',
      email: 'student@example.com',
      password: 'password',
      userType: 'student',
      firstName: 'Student',
      lastName: 'User',
      createdAt: new Date(),
      updatedAt: new Date()
    },
    {
      _id: 'user2',
      email: 'organizer@example.com',
      password: 'password',
      userType: 'organizer',
      firstName: 'Organizer',
      lastName: 'User',
      createdAt: new Date(),
      updatedAt: new Date()
    }
  ],
  questions: [
    {
      _id: 'q1',
      title: 'Two Sum',
      description: 'Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.',
      difficulty: 'easy',
      timeLimit: 30,
      constraints: ['1 <= nums.length <= 10^4', '-10^9 <= nums[i] <= 10^9'],
      tags: ['arrays', 'hash-table'],
      starterCodeJs: 'function twoSum(nums, target) {\n  // Your code here\n}',
      starterCodePython: 'def two_sum(nums, target):\n    # Your code here\n    pass',
      starterCodeJava: 'class Solution {\n    public int[] twoSum(int[] nums, int target) {\n        // Your code here\n        return new int[0];\n    }\n}',
      starterCodeCpp: 'vector<int> twoSum(vector<int>& nums, int target) {\n    // Your code here\n    return {};\n}',
      starterCodeGolang: 'func twoSum(nums []int, target int) []int {\n    // Your code here\n    return nil\n}',
      examples: [{
        input: '[2,7,11,15], 9',
        output: '[0,1]',
        explanation: 'Because nums[0] + nums[1] == 9, we return [0, 1].'
      }],
      createdBy: 'user2',
      createdAt: new Date(),
      updatedAt: new Date()
    },
    {
      _id: 'q2',
      title: 'Add Two Numbers',
      description: 'You are given two non-empty linked lists representing two non-negative integers.',
      difficulty: 'medium',
      timeLimit: 45,
      constraints: ['The number of nodes in each linked list is in the range [1, 100]'],
      tags: ['linked-list', 'math'],
      starterCodeJs: 'function addTwoNumbers(l1, l2) {\n  // Your code here\n}',
      starterCodePython: 'def add_two_numbers(l1, l2):\n    # Your code here\n    pass',
      starterCodeJava: 'class Solution {\n    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {\n        // Your code here\n        return null;\n    }\n}',
      starterCodeCpp: 'ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {\n    // Your code here\n    return nullptr;\n}',
      starterCodeGolang: 'func addTwoNumbers(l1 *ListNode, l2 *ListNode) *ListNode {\n    // Your code here\n    return nil\n}',
      examples: [{
        input: '[2,4,3], [5,6,4]',
        output: '[7,0,8]',
        explanation: '342 + 465 = 807.'
      }],
      createdBy: 'user2',
      createdAt: new Date(),
      updatedAt: new Date()
    }
  ],
  testSessions: [
    {
      _id: 'ts1',
      title: 'Algorithm Assessment',
      description: 'Basic algorithm problems',
      duration: 120,
      startDate: new Date('2023-07-10'),
      endDate: new Date('2023-07-15'),
      enableProctoring: true,
      monitorKeystrokes: true,
      monitorMouseMovements: true,
      preventTabSwitching: true,
      preventCopyPaste: true,
      status: 'active',
      createdBy: 'user2',
      candidates: ['user1'],
      questions: ['q1', 'q2'],
      createdAt: new Date(),
      updatedAt: new Date()
    }
  ],
  userSubmissions: [
    {
      _id: 'sub1',
      testSession: 'ts1',
      question: 'q1',
      user: 'user1',
      code: 'function twoSum(nums, target) {\n  const map = {};\n  for (let i = 0; i < nums.length; i++) {\n    const complement = target - nums[i];\n    if (map[complement] !== undefined) {\n      return [map[complement], i];\n    }\n    map[nums[i]] = i;\n  }\n  return [];\n}',
      language: 'javascript',
      status: 'completed',
      result: 'passed',
      timeSpent: 15,
      completed: true,
      flaggedForCheating: false,
      createdAt: new Date(),
      updatedAt: new Date()
    }
  ]
};

// Authentication Services
export const authenticateUser = async (email: string, password: string) => {
  const isConnected = await connectToDatabase();
  
  if (isConnected) {
    try {
      // In a real application, you would hash the password before comparing
      const user = await User.findOne({ email, password }).lean();
      
      if (!user) {
        throw new Error('Invalid credentials');
      }
      
      return {
        id: user._id.toString(),
        email: user.email,
        userType: user.userType,
        firstName: user.firstName,
        lastName: user.lastName
      };
    } catch (error) {
      console.error('Error authenticating user:', error);
      // Fallback to mock data
      return authenticateWithMockData(email, password);
    }
  } else {
    // Use mock data if not connected
    return authenticateWithMockData(email, password);
  }
};

// Helper function to authenticate with mock data
const authenticateWithMockData = (email: string, password: string) => {
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
  const isConnected = await connectToDatabase();
  
  if (isConnected) {
    try {
      // Check if user already exists
      const existingUser = await User.findOne({ email: userData.email }).lean();
      
      if (existingUser) {
        throw new Error('User already exists');
      }
      
      // Create new user
      const newUser = new User(userData);
      await newUser.save();
      
      return {
        id: newUser._id.toString(),
        email: newUser.email,
        userType: newUser.userType
      };
    } catch (error) {
      console.error('Error registering user:', error);
      // Fallback to mock data
      return registerWithMockData(userData);
    }
  } else {
    // Use mock data if not connected
    return registerWithMockData(userData);
  }
};

// Helper function to register with mock data
const registerWithMockData = (userData: Partial<IUser>) => {
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
  const isConnected = await connectToDatabase();
  
  if (isConnected) {
    try {
      const questions = await Question.find().lean();
      return questions.map(q => ({
        ...q,
        _id: q._id.toString()
      }));
    } catch (error) {
      console.error('Error getting questions:', error);
      return mockData.questions;
    }
  } else {
    return mockData.questions;
  }
};

export const getQuestionById = async (id: string) => {
  const isConnected = await connectToDatabase();
  
  if (isConnected) {
    try {
      const question = await Question.findById(id).lean();
      if (!question) {
        return getQuestionByIdFromMockData(id);
      }
      return {
        ...question,
        _id: question._id.toString()
      };
    } catch (error) {
      console.error('Error getting question by ID:', error);
      return getQuestionByIdFromMockData(id);
    }
  } else {
    return getQuestionByIdFromMockData(id);
  }
};

// Helper function to get question by ID from mock data
const getQuestionByIdFromMockData = (id: string) => {
  return mockData.questions.find(q => q._id === id) || null;
};

// Test Session Services
export const getTestSessions = async (userId: string, userType: string) => {
  const isConnected = await connectToDatabase();
  
  if (isConnected) {
    try {
      let query = {};
      if (userType === 'organizer') {
        query = { createdBy: userId };
      } else {
        query = { candidates: userId, status: 'active' };
      }
      
      const sessions = await TestSession.find(query).lean();
      return sessions.map(session => ({
        ...session,
        _id: session._id.toString(),
        createdBy: session.createdBy.toString(),
        candidates: session.candidates.map((c: any) => c.toString()),
        questions: session.questions.map((q: any) => q.toString())
      }));
    } catch (error) {
      console.error('Error getting test sessions:', error);
      // Fallback to mock data if error
      return getSessionsFromMockData(userId, userType);
    }
  } else {
    return getSessionsFromMockData(userId, userType);
  }
};

// Helper function to get sessions from mock data
const getSessionsFromMockData = (userId: string, userType: string) => {
  if (userType === 'organizer') {
    return mockData.testSessions.filter(session => session.createdBy === userId);
  } else {
    return mockData.testSessions.filter(
      session => session.candidates.includes(userId) && session.status === 'active'
    );
  }
};

export const getTestSessionById = async (id: string) => {
  const isConnected = await connectToDatabase();
  
  if (isConnected) {
    try {
      // Get the session
      const session = await TestSession.findById(id).lean();
      
      if (!session) {
        return getSessionByIdFromMockData(id);
      }
      
      // Get the questions if needed
      let questionDetails = [];
      if (session.questions && session.questions.length > 0) {
        const questionIds = session.questions.map(qId => qId.toString());
        questionDetails = await Question.find({ 
          _id: { $in: questionIds } 
        }).lean();
        
        questionDetails = questionDetails.map(q => ({
          ...q,
          _id: q._id.toString()
        }));
      }
      
      return {
        ...session,
        _id: session._id.toString(),
        createdBy: session.createdBy.toString(),
        candidates: session.candidates.map((c: any) => c.toString()),
        questions: questionDetails.length > 0 ? questionDetails : 
                  session.questions.map((q: any) => q.toString())
      };
    } catch (error) {
      console.error('Error getting test session by ID:', error);
      return getSessionByIdFromMockData(id);
    }
  } else {
    return getSessionByIdFromMockData(id);
  }
};

// Helper function to get session by ID from mock data
const getSessionByIdFromMockData = (id: string) => {
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

// User Submission Services - Simplified
export const getUserSubmissions = async (userId: string, testSessionId: string) => {
  return getUserSubmissionsFromMockData(userId, testSessionId);
};

// Helper function to get user submissions from mock data
const getUserSubmissionsFromMockData = (userId: string, testSessionId: string) => {
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
  return saveSubmissionToMockData(submissionData);
};

// Helper function to save submission to mock data
const saveSubmissionToMockData = (submissionData: Partial<IUserSubmission>) => {
  // Convert ObjectIds to strings for comparison
  const userStr = submissionData.user?.toString();
  const questionStr = submissionData.question?.toString();
  const testSessionStr = submissionData.testSession?.toString();
  
  // Check if a mock submission already exists
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

// Analytics for Organizer Dashboard - Simplified to use mock data only
export const getOrganizerInsights = async (organizerId: string) => {
  return getMockOrganizerInsights(organizerId);
};

// Helper function to get mock organizer insights
const getMockOrganizerInsights = (organizerId: string) => {
  const testSessions = mockData.testSessions.filter(session => session.createdBy === organizerId);
  const sessionIds = testSessions.map(session => session._id);
  
  const submissions = mockData.userSubmissions.filter(
    sub => sessionIds.includes(sub.testSession as string)
  );
  
  // Calculate stats from mock data
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
