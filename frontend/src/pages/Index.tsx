
import React, { useState } from 'react';
import QuestionSidebar from '@/components/QuestionSidebar';
import QuestionDisplay from '@/components/QuestionDisplay';
import CodeEditor from '@/components/CodeEditor';
import { toast } from "@/hooks/use-toast";

// Mock data for questions
const mockQuestions = [
  {
    id: 1,
    title: "Two Sum",
    description: "Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.\n\nYou may assume that each input would have exactly one solution, and you may not use the same element twice.\n\nYou can return the answer in any order.",
    examples: [
      {
        input: "nums = [2,7,11,15], target = 9",
        output: "[0,1]",
        explanation: "Because nums[0] + nums[1] == 9, we return [0, 1]."
      },
      {
        input: "nums = [3,2,4], target = 6",
        output: "[1,2]"
      },
      {
        input: "nums = [3,3], target = 6",
        output: "[0,1]"
      }
    ],
    constraints: [
      "2 <= nums.length <= 10^4",
      "-10^9 <= nums[i] <= 10^9",
      "-10^9 <= target <= 10^9",
      "Only one valid answer exists."
    ],
    difficulty: "easy" as const,
    tags: ["Array", "Hash Table"],
    timeComplexity: "Expected Time: O(n)",
    spaceComplexity: "Expected Space: O(n)",
    completed: false
  },
  {
    id: 2,
    title: "Add Two Numbers",
    description: "You are given two non-empty linked lists representing two non-negative integers. The digits are stored in reverse order, and each of their nodes contains a single digit. Add the two numbers and return the sum as a linked list.\n\nYou may assume the two numbers do not contain any leading zero, except the number 0 itself.",
    examples: [
      {
        input: "l1 = [2,4,3], l2 = [5,6,4]",
        output: "[7,0,8]",
        explanation: "342 + 465 = 807."
      },
      {
        input: "l1 = [0], l2 = [0]",
        output: "[0]"
      },
      {
        input: "l1 = [9,9,9,9,9,9,9], l2 = [9,9,9,9]",
        output: "[8,9,9,9,0,0,0,1]"
      }
    ],
    constraints: [
      "The number of nodes in each linked list is in the range [1, 100].",
      "0 <= Node.val <= 9",
      "It is guaranteed that the list represents a number that does not have leading zeros."
    ],
    difficulty: "medium" as const,
    tags: ["Linked List", "Math", "Recursion"],
    completed: false
  },
  {
    id: 3,
    title: "Median of Two Sorted Arrays",
    description: "Given two sorted arrays nums1 and nums2 of size m and n respectively, return the median of the two sorted arrays.\n\nThe overall run time complexity should be O(log (m+n)).",
    examples: [
      {
        input: "nums1 = [1,3], nums2 = [2]",
        output: "2.00000",
        explanation: "merged array = [1,2,3] and median is 2."
      },
      {
        input: "nums1 = [1,2], nums2 = [3,4]",
        output: "2.50000",
        explanation: "merged array = [1,2,3,4] and median is (2 + 3) / 2 = 2.5."
      }
    ],
    constraints: [
      "nums1.length == m",
      "nums2.length == n",
      "0 <= m <= 1000",
      "0 <= n <= 1000",
      "1 <= m + n <= 2000",
      "-10^6 <= nums1[i], nums2[i] <= 10^6"
    ],
    difficulty: "hard" as const,
    tags: ["Array", "Binary Search", "Divide and Conquer"],
    completed: false
  },
  {
    id: 4,
    title: "Longest Palindromic Substring",
    description: "Given a string s, return the longest palindromic substring in s.",
    examples: [
      {
        input: "s = \"babad\"",
        output: "\"bab\"",
        explanation: "\"aba\" is also a valid answer."
      },
      {
        input: "s = \"cbbd\"",
        output: "\"bb\""
      }
    ],
    constraints: [
      "1 <= s.length <= 1000",
      "s consist of only digits and English letters."
    ],
    difficulty: "medium" as const,
    tags: ["String", "Dynamic Programming"],
    completed: false
  },
  {
    id: 5,
    title: "ZigZag Conversion",
    description: "The string \"PAYPALISHIRING\" is written in a zigzag pattern on a given number of rows like this: (you may want to display this pattern in a fixed font for better legibility)\n\nP   A   H   N\nA P L S I I G\nY   I   R\n\nAnd then read line by line: \"PAHNAPLSIIGYIR\"\n\nWrite the code that will take a string and make this conversion given a number of rows.",
    examples: [
      {
        input: "s = \"PAYPALISHIRING\", numRows = 3",
        output: "\"PAHNAPLSIIGYIR\""
      },
      {
        input: "s = \"PAYPALISHIRING\", numRows = 4",
        output: "\"PINALSIGYAHRPI\"",
        explanation: "P     I    N\nA   L S  I G\nY A   H R\nP     I"
      }
    ],
    constraints: [
      "1 <= s.length <= 1000",
      "s consists of English letters (lower-case and upper-case), ',' and '.'.",
      "1 <= numRows <= 1000"
    ],
    difficulty: "medium" as const,
    tags: ["String"],
    completed: false
  }
];

const getLanguageStarterCode = (language: string) => {
  switch (language) {
    case 'javascript':
      return `/**
 * @param {number[]} nums
 * @param {number} target
 * @return {number[]}
 */
function twoSum(nums, target) {
    // Your code here
    
};`;
    case 'python':
      return `class Solution:
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        # Your code here
        pass`;
    default:
      return `// Your solution here`;
  }
};

const Index = () => {
  const [currentQuestionId, setCurrentQuestionId] = useState(1);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [questions, setQuestions] = useState(mockQuestions);
  
  const currentQuestion = questions.find(q => q.id === currentQuestionId) || questions[0];
  
  const handleSelectQuestion = (questionId: number) => {
    setCurrentQuestionId(questionId);
  };
  
  const handlePrevQuestion = () => {
    const index = questions.findIndex(q => q.id === currentQuestionId);
    if (index > 0) {
      setCurrentQuestionId(questions[index - 1].id);
    }
  };
  
  const handleNextQuestion = () => {
    const index = questions.findIndex(q => q.id === currentQuestionId);
    if (index < questions.length - 1) {
      setCurrentQuestionId(questions[index + 1].id);
    }
  };
  
  const handleSubmitCode = (code: string, language: string) => {
    console.log(`Submitting ${language} code for question ${currentQuestionId}:`, code);
    
    // Mark current question as completed
    const updatedQuestions = questions.map(q => 
      q.id === currentQuestionId ? { ...q, completed: true } : q
    );
    setQuestions(updatedQuestions);
    
    // Show success toast
    toast({
      title: "Solution submitted",
      description: "Your solution has been submitted successfully",
      duration: 3000
    });
  };
  
  return (
    <div className={`flex flex-row h-screen overflow-hidden transition-all duration-300 ease-in-out`}>
      <QuestionSidebar
        questions={questions}
        currentQuestionId={currentQuestionId}
        onSelectQuestion={handleSelectQuestion}
        collapsed={sidebarCollapsed}
        onToggleCollapse={() => setSidebarCollapsed(!sidebarCollapsed)}
      />
      
      <div className={`flex-1 flex ${sidebarCollapsed ? 'pl-[60px]' : 'pl-[280px]'} absolute inset-y-0 left-0 right-0 transition-all duration-300 ease-in-out`}>
        <div className="w-1/2 border-r border-assessment-border overflow-y-auto">
          <QuestionDisplay
            question={currentQuestion}
            onPrevQuestion={handlePrevQuestion}
            onNextQuestion={handleNextQuestion}
            hasPrevQuestion={questions.findIndex(q => q.id === currentQuestionId) > 0}
            hasNextQuestion={questions.findIndex(q => q.id === currentQuestionId) < questions.length - 1}
          />
        </div>
        
        <div className="w-1/2 overflow-hidden">
          <CodeEditor
            questionId={currentQuestionId}
            initialCode={getLanguageStarterCode('javascript')}
            onSubmit={handleSubmitCode}
          />
        </div>
      </div>
    </div>
  );
};

export default Index;
