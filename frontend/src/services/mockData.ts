// src/services/mockData.ts
export const mockData = {
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
  
  export default mockData;