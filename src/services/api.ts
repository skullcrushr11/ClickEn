
import { connectToDatabase, isMongoConnected } from '../config/mongodb';
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
      // For simplicity, we're doing a direct comparison here
      const user = await User.findOne({ email, password }).lean().exec();
      
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
    } catch (error) {
      console.error('Error authenticating user:', error);
      throw error;
    }
  } else {
    // Use mock data if not connected
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
  }
};

export const registerUser = async (userData: Partial<IUser>) => {
  const isConnected = await connectToDatabase();
  
  if (isConnected) {
    try {
      // Check if user already exists
      const existingUser = await User.findOne({ email: userData.email }).lean().exec();
      
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
    } catch (error) {
      console.error('Error registering user:', error);
      throw error;
    }
  } else {
    // Use mock data if not connected
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
  }
};

// Question Services
export const getQuestions = async () => {
  const isConnected = await connectToDatabase();
  
  if (isConnected) {
    try {
      return await Question.find().lean().exec();
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
      return await Question.findById(id).lean().exec();
    } catch (error) {
      console.error('Error getting question by ID:', error);
      return mockData.questions.find(q => q._id === id) || null;
    }
  } else {
    return mockData.questions.find(q => q._id === id) || null;
  }
};

export const createQuestion = async (questionData: Partial<IQuestion>) => {
  const isConnected = await connectToDatabase();
  
  if (isConnected) {
    try {
      const newQuestion = new Question(questionData);
      await newQuestion.save();
      return newQuestion;
    } catch (error) {
      console.error('Error creating question:', error);
      throw error;
    }
  } else {
    const newQuestion = {
      _id: `q${mockData.questions.length + 1}`,
      ...questionData,
      createdAt: new Date(),
      updatedAt: new Date()
    };
    
    mockData.questions.push(newQuestion as any);
    return newQuestion;
  }
};

// Test Session Services
export const getTestSessions = async (userId: string, userType: string) => {
  const isConnected = await connectToDatabase();
  
  if (isConnected) {
    try {
      if (userType === 'organizer') {
        return await TestSession.find({ createdBy: userId }).lean().exec();
      } else {
        // For students, find test sessions they are candidates for
        return await TestSession.find({ candidates: userId, status: 'active' }).lean().exec();
      }
    } catch (error) {
      console.error('Error getting test sessions:', error);
      // Fallback to mock data if error
      if (userType === 'organizer') {
        return mockData.testSessions.filter(session => session.createdBy === userId);
      } else {
        return mockData.testSessions.filter(
          session => session.candidates.includes(userId) && session.status === 'active'
        );
      }
    }
  } else {
    if (userType === 'organizer') {
      return mockData.testSessions.filter(session => session.createdBy === userId);
    } else {
      return mockData.testSessions.filter(
        session => session.candidates.includes(userId) && session.status === 'active'
      );
    }
  }
};

export const getTestSessionById = async (id: string) => {
  const isConnected = await connectToDatabase();
  
  if (isConnected) {
    try {
      return await TestSession.findById(id).populate('questions').lean().exec();
    } catch (error) {
      console.error('Error getting test session by ID:', error);
      // Fallback to mock data if error
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
    }
  } else {
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
  }
};

export const createTestSession = async (sessionData: Partial<ITestSession>) => {
  const isConnected = await connectToDatabase();
  
  if (isConnected) {
    try {
      const newSession = new TestSession(sessionData);
      await newSession.save();
      return newSession;
    } catch (error) {
      console.error('Error creating test session:', error);
      throw error;
    }
  } else {
    const newSession = {
      _id: `ts${mockData.testSessions.length + 1}`,
      ...sessionData,
      createdAt: new Date(),
      updatedAt: new Date()
    };
    
    mockData.testSessions.push(newSession as any);
    return newSession;
  }
};

// User Submission Services
export const getUserSubmissions = async (userId: string, testSessionId: string) => {
  const isConnected = await connectToDatabase();
  
  if (isConnected) {
    try {
      return await UserSubmission.find({ 
        user: userId,
        testSession: testSessionId 
      }).populate('question').lean().exec();
    } catch (error) {
      console.error('Error getting user submissions:', error);
      // Fallback to mock data if error
      const submissions = mockData.userSubmissions.filter(
        sub => sub.user === userId && sub.testSession === testSessionId
      );
      
      // Simulate populated questions
      return submissions.map(sub => ({
        ...sub,
        question: mockData.questions.find(q => q._id === sub.question)
      }));
    }
  } else {
    const submissions = mockData.userSubmissions.filter(
      sub => sub.user === userId && sub.testSession === testSessionId
    );
    
    // Simulate populated questions
    return submissions.map(sub => ({
      ...sub,
      question: mockData.questions.find(q => q._id === sub.question)
    }));
  }
};

export const saveUserSubmission = async (submissionData: Partial<IUserSubmission>) => {
  const isConnected = await connectToDatabase();
  
  if (isConnected) {
    try {
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
    } catch (error) {
      console.error('Error saving user submission:', error);
      throw error;
    }
  } else {
    // Check if a mock submission already exists
    const existingIndex = mockData.userSubmissions.findIndex(
      sub => sub.user === submissionData.user && 
             sub.question === submissionData.question && 
             sub.testSession === submissionData.testSession
    );
    
    if (existingIndex !== -1) {
      // Update existing submission
      mockData.userSubmissions[existingIndex] = {
        ...mockData.userSubmissions[existingIndex],
        ...submissionData,
        updatedAt: new Date()
      };
      return mockData.userSubmissions[existingIndex];
    } else {
      // Create new submission
      const newSubmission = {
        _id: `sub${mockData.userSubmissions.length + 1}`,
        ...submissionData,
        createdAt: new Date(),
        updatedAt: new Date()
      };
      
      mockData.userSubmissions.push(newSubmission as any);
      return newSubmission;
    }
  }
};

// Analytics for Organizer Dashboard
export const getOrganizerInsights = async (organizerId: string) => {
  const isConnected = await connectToDatabase();
  
  if (isConnected) {
    try {
      const testSessions = await TestSession.find({ createdBy: organizerId }).lean().exec();
      
      const sessionIds = testSessions.map(session => session._id);
      
      const submissions = await UserSubmission.find({
        testSession: { $in: sessionIds }
      }).lean().exec();
      
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
    } catch (error) {
      console.error('Error getting organizer insights:', error);
      // Fallback to mock data
      return getMockOrganizerInsights(organizerId);
    }
  } else {
    // Use mock data for analytics
    return getMockOrganizerInsights(organizerId);
  }
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
