
import React, { useState, useEffect, useCallback } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import QuestionSidebar from '@/components/QuestionSidebar';
import QuestionDisplay from '@/components/QuestionDisplay';
import CodeEditor from '@/components/CodeEditor';
import { useToast } from '@/hooks/use-toast';
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { AlertCircle } from "lucide-react";
import { initProctoring, captureScreenshot } from '@/utils/proctoring';

interface Question {
  id: number;
  title: string;
  description: string;
  examples: Array<{
    input: string;
    output: string;
    explanation?: string;
  }>;
  constraints: string[];
  difficulty: 'easy' | 'medium' | 'hard';
  tags: string[];
  completed: boolean;
  timeComplexity?: string;
  spaceComplexity?: string;
}

const TestSession: React.FC = () => {
  const { testId } = useParams<{ testId: string }>();
  const navigate = useNavigate();
  const { toast } = useToast();
  
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [currentQuestionId, setCurrentQuestionId] = useState(1);
  const [questions, setQuestions] = useState<Question[]>([]);
  const [timeLeft, setTimeLeft] = useState(60 * 60); // 60 minutes in seconds
  const [showWarning, setShowWarning] = useState(false);
  const [warningMessage, setWarningMessage] = useState('');
  const [suspiciousEvents, setSuspiciousEvents] = useState<any[]>([]);
  const [confirmEndDialog, setConfirmEndDialog] = useState(false);
  
  // Mock data for questions
  useEffect(() => {
    // In a real app, this would fetch from an API based on testId
    setQuestions([
      {
        id: 1,
        title: "Two Sum",
        difficulty: "easy",
        description: `Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.

You may assume that each input would have exactly one solution, and you may not use the same element twice.

You can return the answer in any order.`,
        examples: [
          {
            input: "nums = [2,7,11,15], target = 9",
            output: "[0,1]",
            explanation: "Because nums[0] + nums[1] == 9, we return [0, 1]."
          },
          {
            input: "nums = [3,2,4], target = 6",
            output: "[1,2]"
          }
        ],
        constraints: [
          "2 <= nums.length <= 10^4",
          "-10^9 <= nums[i] <= 10^9",
          "-10^9 <= target <= 10^9",
          "Only one valid answer exists."
        ],
        tags: ["arrays", "hash-table"],
        completed: false
      },
      {
        id: 2,
        title: "Palindrome Number",
        difficulty: "easy",
        description: `Given an integer x, return true if x is a palindrome, and false otherwise.

An integer is a palindrome when it reads the same forward and backward.

For example, 121 is a palindrome while 123 is not.`,
        examples: [
          {
            input: "x = 121",
            output: "true",
            explanation: "121 reads as 121 from left to right and from right to left."
          },
          {
            input: "x = -121",
            output: "false",
            explanation: "From left to right, it reads -121. From right to left, it reads 121-. Therefore it is not a palindrome."
          }
        ],
        constraints: [
          "-2^31 <= x <= 2^31 - 1"
        ],
        tags: ["math"],
        completed: false
      },
      {
        id: 3,
        title: "Valid Parentheses",
        difficulty: "medium",
        description: `Given a string s containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid.

An input string is valid if:
1. Open brackets must be closed by the same type of brackets.
2. Open brackets must be closed in the correct order.`,
        examples: [
          {
            input: "s = '()'",
            output: "true"
          },
          {
            input: "s = '()[]{}'",
            output: "true"
          },
          {
            input: "s = '(]'",
            output: "false"
          }
        ],
        constraints: [
          "1 <= s.length <= 10^4",
          "s consists of parentheses only '()[]{}'."
        ],
        tags: ["stack", "string"],
        completed: false
      },
      {
        id: 4,
        title: "Merge Two Sorted Lists",
        difficulty: "easy",
        description: `You are given the heads of two sorted linked lists list1 and list2.

Merge the two lists in a one sorted list. The list should be made by splicing together the nodes of the first two lists.

Return the head of the merged linked list.`,
        examples: [
          {
            input: "list1 = [1,2,4], list2 = [1,3,4]",
            output: "[1,1,2,3,4,4]"
          },
          {
            input: "list1 = [], list2 = []",
            output: "[]"
          },
          {
            input: "list1 = [], list2 = [0]",
            output: "[0]"
          }
        ],
        constraints: [
          "The number of nodes in both lists is in the range [0, 50].",
          "-100 <= Node.val <= 100",
          "Both list1 and list2 are sorted in non-decreasing order."
        ],
        tags: ["linked-list", "recursion"],
        completed: false
      },
      {
        id: 5,
        title: "Maximum Subarray",
        difficulty: "medium",
        description: `Given an integer array nums, find the subarray with the largest sum, and return its sum.

A subarray is a contiguous non-empty sequence of elements within an array.`,
        examples: [
          {
            input: "nums = [-2,1,-3,4,-1,2,1,-5,4]",
            output: "6",
            explanation: "The subarray [4,-1,2,1] has the largest sum 6."
          },
          {
            input: "nums = [1]",
            output: "1"
          },
          {
            input: "nums = [5,4,-1,7,8]",
            output: "23"
          }
        ],
        constraints: [
          "1 <= nums.length <= 10^5",
          "-10^4 <= nums[i] <= 10^4"
        ],
        tags: ["array", "divide-and-conquer", "dynamic-programming"],
        completed: false
      }
    ]);
  }, [testId]);
  
  // Initialize proctoring
  useEffect(() => {
    const handleSuspiciousActivity = useCallback(async (eventType: string, data: any) => {
      console.log('Suspicious activity detected:', eventType, data);
      
      // Take a screenshot
      const screenshot = await captureScreenshot();
      
      // Add to suspicious events
      setSuspiciousEvents(prev => [
        ...prev, 
        { 
          type: eventType, 
          data, 
          timestamp: new Date(), 
          screenshot 
        }
      ]);
      
      // Show warning to user
      let message = 'Suspicious activity detected.';
      switch (eventType) {
        case 'tab_switch':
          message = 'Warning: Tab switching detected! This will be reported.';
          break;
        case 'copy_paste':
          message = 'Copy/paste is not allowed during this assessment!';
          break;
        case 'irregular_keystrokes':
          message = 'Unusual typing pattern detected.';
          break;
        case 'irregular_mouse':
          message = 'Unusual mouse movement detected.';
          break;
      }
      
      setWarningMessage(message);
      setShowWarning(true);
      
      // Hide warning after 5 seconds
      setTimeout(() => {
        setShowWarning(false);
      }, 5000);
      
      // Report to server (in a real app)
      // This would send the event to a backend to alert proctors
      console.log('Reporting suspicious activity to server:', {
        testId,
        studentId: localStorage.getItem('userEmail'),
        eventType,
        data,
        timestamp: new Date(),
        screenshot
      });
    }, [testId]);
    
    const { stopProctoring } = initProctoring({
      monitorKeystrokes: true,
      monitorMouseMovements: true,
      preventTabSwitching: true,
      preventCopyPaste: true,
      onSuspiciousActivity: handleSuspiciousActivity
    });
    
    // Timer for the test
    const timer = setInterval(() => {
      setTimeLeft(prev => {
        if (prev <= 0) {
          clearInterval(timer);
          handleEndTest();
          return 0;
        }
        return prev - 1;
      });
    }, 1000);
    
    // Cleanup
    return () => {
      stopProctoring();
      clearInterval(timer);
    };
  }, [testId]);
  
  const currentQuestion = questions.find(q => q.id === currentQuestionId) || questions[0];
  
  const handleSelectQuestion = (questionId: number) => {
    setCurrentQuestionId(questionId);
  };
  
  const handlePrevQuestion = () => {
    const currentIndex = questions.findIndex(q => q.id === currentQuestionId);
    if (currentIndex > 0) {
      setCurrentQuestionId(questions[currentIndex - 1].id);
    }
  };
  
  const handleNextQuestion = () => {
    const currentIndex = questions.findIndex(q => q.id === currentQuestionId);
    if (currentIndex < questions.length - 1) {
      setCurrentQuestionId(questions[currentIndex + 1].id);
    }
  };
  
  const handleSubmitCode = (code: string, language: string) => {
    console.log(`Code submitted for question ${currentQuestionId}:`, { code, language });
    
    // Mark the question as completed
    setQuestions(
      questions.map(q => 
        q.id === currentQuestionId ? { ...q, completed: true } : q
      )
    );
    
    toast({
      title: "Solution submitted",
      description: "Your solution has been successfully submitted."
    });
    
    // Move to the next question if available
    const currentIndex = questions.findIndex(q => q.id === currentQuestionId);
    if (currentIndex < questions.length - 1) {
      setCurrentQuestionId(questions[currentIndex + 1].id);
    }
  };
  
  const getLanguageStarterCode = (language: string) => {
    // This would normally fetch from the question data
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
      default:
        return '// Your solution here';
    }
  };
  
  const formatTime = (seconds: number) => {
    const hrs = Math.floor(seconds / 3600);
    const mins = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;
    
    return `${hrs.toString().padStart(2, '0')}:${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };
  
  const handleEndTest = () => {
    // In a real app, this would submit all solutions and end the test
    navigate('/student-dashboard');
    toast({
      title: "Test completed",
      description: "Your solutions have been submitted successfully."
    });
  };
  
  return (
    <div className="flex flex-row h-screen overflow-hidden transition-all duration-300 ease-in-out">
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
      
      {/* Proctoring warning */}
      {showWarning && (
        <Alert 
          variant="destructive" 
          className="fixed bottom-4 right-4 w-96 shadow-lg border-l-4 border-destructive"
        >
          <AlertCircle className="h-4 w-4" />
          <AlertTitle>Proctoring Alert</AlertTitle>
          <AlertDescription>
            {warningMessage}
          </AlertDescription>
        </Alert>
      )}
      
      {/* Timer display */}
      <div className="fixed top-4 right-4 bg-white shadow-md rounded-md px-4 py-2 flex items-center space-x-2 border">
        <Clock className="h-4 w-4 text-assessment-primary" />
        <span className={`font-mono font-medium ${timeLeft < 300 ? 'text-destructive animate-pulse' : ''}`}>
          {formatTime(timeLeft)}
        </span>
        <Button 
          variant="outline" 
          size="sm"
          onClick={() => setConfirmEndDialog(true)}
          className="ml-2 h-7 text-xs"
        >
          End Test
        </Button>
      </div>
      
      {/* Confirm end test dialog */}
      <Dialog open={confirmEndDialog} onOpenChange={setConfirmEndDialog}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>End Test Early?</DialogTitle>
            <DialogDescription>
              Are you sure you want to end the test? You still have time remaining, and all unanswered questions will be marked as incomplete.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="outline" onClick={() => setConfirmEndDialog(false)}>
              Continue Testing
            </Button>
            <Button variant="destructive" onClick={handleEndTest}>
              End Test
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
};

export default TestSession;
