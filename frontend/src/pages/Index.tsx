import React, { useState, useEffect, useRef,useCallback,CSSProperties } from 'react';
import QuestionSidebar from '@/components/QuestionSidebar';
import QuestionDisplay from '@/components/QuestionDisplay';
import CodeEditor from '@/components/CodeEditor';
import { toast } from "@/hooks/use-toast";
import { goFullScreen, handleCopy, handleCut, handlePaste } from '@/utils/coding-file';
import { KeyboardEventStreamer } from '@/utils/event-emitter';


const mockQuestions = [
  {
    id: 1,
    title: "Find Maximum",
    type: "coding",
    description: "Given an array of integers, return the maximum number in the array.",
    examples: [
      {
        input: "nums = [1, 5, 3, 9, 2]",
        output: "9"
      },
      {
        input: "nums = [-1, -5, -3]",
        output: "-1"
      }
    ],
    constraints: [
      "1 <= nums.length <= 100",
      "-1000 <= nums[i] <= 1000"
    ],
    difficulty: "easy" as const,
    tags: ["Array", "Math"],
    timeComplexity: "Expected Time: O(n)",
    spaceComplexity: "Expected Space: O(1)",
    completed: false
  },
  {
    id: 2,
    title: "Sum of Digits",
    type: "coding",
    description: "Given a non-negative integer n, return the sum of its digits.",
    examples: [
      {
        input: "n = 123",
        output: "6",
        explanation: "1 + 2 + 3 = 6"
      },
      {
        input: "n = 9876",
        output: "30"
      }
    ],
    constraints: [
      "0 <= n <= 10^6"
    ],
    difficulty: "easy" as const,
    tags: ["Math"],
    timeComplexity: "Expected Time: O(log n)",
    spaceComplexity: "Expected Space: O(1)",
    completed: false
  },
  {
    id: 3,
    title: "Find Middle Element",
    type: "mcq",
    description: "What is the middle element of the list nums = [2, 4, 6, 8, 10]?",
    examples: [
      {
        input: "nums = [2, 4, 6, 8, 10]",
        output: "6",
        explanation: "The middle element is at index 2 (0-based)."
      }
    ],
    constraints: [
      "The list has an odd length."
    ],
    difficulty: "easy" as const,
    tags: ["Array"],
    options: [
      { text: "4", isCorrect: false },
      { text: "6", isCorrect: true },
      { text: "8", isCorrect: false },
      { text: "10", isCorrect: false }
    ],
    completed: false
  },
  {
    id: 4,
    title: "Reverse a String",
    type: "subjective",
    description: "Given a string s, return the reversed version of the string.",
    examples: [
      {
        input: "s = \"hello\"",
        output: "\"olleh\""
      },
      {
        input: "s = \"world\"",
        output: "\"dlrow\""
      }
    ],
    constraints: [
      "1 <= s.length <= 100",
      "s consists of only lowercase English letters."
    ],
    difficulty: "easy" as const,
    tags: ["String"],
    completed: false
  },
  {
    id: 5,
    title: "Time Complexity",
    type: "mcq",
    description: "What is the time complexity of binary search algorithm?",
    examples: [
      {
        input: "Binary search on a sorted array",
        output: "O(log n)",
        explanation: "The search space is halved in each step."
      }
    ],
    constraints: [],
    difficulty: "medium" as const,
    tags: ["Algorithm", "Complexity"],
    options: [
      { text: "O(1)", isCorrect: false },
      { text: "O(n)", isCorrect: false },
      { text: "O(log n)", isCorrect: true },
      { text: "O(n log n)", isCorrect: false }
    ],
    completed: false
  },
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
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [hasInteracted, setHasInteracted] = useState(false);
  const [internalClipboard, setInternalClipboard] = useState<string>("");


  useEffect(() => {
    const fullscreenPreferred = localStorage.getItem('fullscreenPreferred') === 'true';
    
    // Track if we're in programmatic fullscreen (via API)
    const isInProgrammaticFullscreen = () => Boolean(document.fullscreenElement);
    
    // Track browser dimensions to detect F11 fullscreen
    const isLikelyInF11Fullscreen = () => {
      return window.innerWidth === screen.width && window.innerHeight === screen.height;
    };
    
    // Combined check for any type of fullscreen
    const checkFullscreenState = () => {
      const inFullscreen = isInProgrammaticFullscreen() || isLikelyInF11Fullscreen();
      
      // Only update if state changed to prevent loops
      if (inFullscreen !== isFullscreen) {
        setIsFullscreen(inFullscreen);
        
        // When exiting fullscreen, force question blurring
        if (!inFullscreen) {
          setIsQuestionBlurred(true);
          
          if (hasInteracted) {
            localStorage.setItem('fullscreenPreferred', 'false');
          }
        }
      }
    };
  
    // Handle standard fullscreen API events
    const handleFullscreenChange = () => {
      checkFullscreenState();
    };
    
    // Handle resize events (for F11 detection)
    const handleResize = () => {
      // Small delay to ensure accurate dimensions after resize completes
      setTimeout(checkFullscreenState, 100);
    };
  
    // Attach multiple event listeners to catch all fullscreen changes
    document.addEventListener("fullscreenchange", handleFullscreenChange);
    document.addEventListener("webkitfullscreenchange", handleFullscreenChange);
    document.addEventListener("mozfullscreenchange", handleFullscreenChange);
    document.addEventListener("MSFullscreenChange", handleFullscreenChange);
    
    // Add resize listener for F11 detection
    window.addEventListener("resize", handleResize);
  
    const attemptFullscreenOnLoad = () => {
      if (fullscreenPreferred) {
        setTimeout(() => {
          goFullScreen(setIsFullscreen);
        }, 1000);
      }
    };
  
    // Handle first-time interaction
    const handleUserInteraction = () => {
      setHasInteracted(true);
      goFullScreen(setIsFullscreen);
      document.removeEventListener("click", handleUserInteraction);
      localStorage.setItem('fullscreenPreferred', 'true');
    };
  
    // Check initial state
    checkFullscreenState();
  
    // If user hasn't interacted yet, set up the click listener
    if (!hasInteracted) {
      document.addEventListener("click", handleUserInteraction);
    } else {
      // If they have interacted before (e.g., on a refresh), try to restore fullscreen
      attemptFullscreenOnLoad();
    }
  
    return () => {
      document.removeEventListener("fullscreenchange", handleFullscreenChange);
      document.removeEventListener("webkitfullscreenchange", handleFullscreenChange);
      document.removeEventListener("mozfullscreenchange", handleFullscreenChange);
      document.removeEventListener("MSFullscreenChange", handleFullscreenChange);
      window.removeEventListener("resize", handleResize);
      document.removeEventListener("click", handleUserInteraction);
    };
  }, [hasInteracted, isFullscreen]);

  const keyEventStreamerRef = useRef<KeyboardEventStreamer | null>(null);

  useEffect(() => {
    // Add event listeners when component mounts
    const copyHandler = (e: ClipboardEvent) => handleCopy(e, setInternalClipboard);
    const cutHandler = (e: ClipboardEvent) => handleCut(e, setInternalClipboard);
    const pasteHandler = (e: ClipboardEvent) => handlePaste(e, internalClipboard);

    keyEventStreamerRef.current = new KeyboardEventStreamer('http://localhost:5000')

    const keyDownHandler = (e: KeyboardEvent) => {
      console.log('Key down:', e.key);
      keyEventStreamerRef.current?.sendEvent('KD', e.key);
    };

    const keyUpHandler = (e: KeyboardEvent) => {
      console.log('Key up:', e.key);
      keyEventStreamerRef.current?.sendEvent('KU', e.key);
    };

    // Attach the event listeners to the document
    document.addEventListener('copy', copyHandler as EventListener);
    document.addEventListener('cut', cutHandler as EventListener);
    document.addEventListener('paste', pasteHandler as EventListener);
    document.addEventListener('keydown', keyDownHandler as EventListener);
    document.addEventListener('keyup', keyUpHandler as EventListener);


    // Clean up function to remove event listeners when component unmounts
    return () => {
      document.removeEventListener('copy', copyHandler as EventListener);
      document.removeEventListener('cut', cutHandler as EventListener);
      document.removeEventListener('paste', pasteHandler as EventListener);
      document.removeEventListener('keydown', keyDownHandler as EventListener);
      document.removeEventListener('keyup', keyUpHandler as EventListener);
      keyEventStreamerRef.current?.disconnect();
    };
  }, [internalClipboard]);

  const currentQuestion = questions.find(q => q.id === currentQuestionId) || questions[0];

  const [isQuestionBlurred, setIsQuestionBlurred] = useState(true); // Start blurred
  const [isQuestionHovered, setIsQuestionHovered] = useState(false);
  const [isKeyPressed, setIsKeyPressed] = useState(false);
  const [countdownValue, setCountdownValue] = useState<number | null>(null);
  
  const mousePosRef = useRef<{x: number, y: number}>({ x: 0, y: 0 });
  const countdownTimerRef = useRef<NodeJS.Timeout | null>(null);
  const questionSectionRef = useRef<HTMLDivElement>(null);
  const lastActivityTimeRef = useRef<number>(Date.now());
  const activityCheckIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const inactivityTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  
  // Flag to track if blur was due to inactivity
  const blurReasonRef = useRef<'inactivity' | 'manual'>('manual');
  
  // Function to check for mouse inactivity
  const checkMouseActivity = useCallback(() => {
    const now = Date.now();
    const timeSinceLastActivity = now - lastActivityTimeRef.current;
    
    // If mouse hasn't moved for more than 1 second and the question is not blurred
    if (timeSinceLastActivity > 1000 && !isQuestionBlurred && isKeyPressed && isQuestionHovered) {
      // Set a timeout before starting countdown if not already waiting
      if (inactivityTimeoutRef.current === null && countdownValue === null) {
        inactivityTimeoutRef.current = setTimeout(() => {
          // Start countdown after 1-second delay
          startCountdown();
          inactivityTimeoutRef.current = null;
        }, 1000);
      }
    }
  }, [countdownValue, isQuestionBlurred, isKeyPressed, isQuestionHovered]);
  
  // Function to start the countdown timer
  const startCountdown = useCallback(() => {
    // Start at 3 seconds
    setCountdownValue(3);
    
    const runCountdown = () => {
      setCountdownValue((prev) => {
        // If somehow the countdown is null, end it
        if (prev === null) return null;
        
        const newValue = prev - 1;
        
        // When countdown reaches 0, blur the question and reset
        if (newValue <= 0) {
          setIsQuestionBlurred(true);
          blurReasonRef.current = 'inactivity'; // Mark blur as due to inactivity
          
          // Clear the countdown timer
          if (countdownTimerRef.current) {
            clearTimeout(countdownTimerRef.current);
            countdownTimerRef.current = null;
          }
          
          return null;
        }
        
        // Continue countdown
        countdownTimerRef.current = setTimeout(runCountdown, 1000);
        return newValue;
      });
    };
    
    // Clear any existing countdown
    if (countdownTimerRef.current) {
      clearTimeout(countdownTimerRef.current);
    }
    
    // Start the countdown
    countdownTimerRef.current = setTimeout(runCountdown, 1000);
  }, []);
  
  // Start or stop the activity check interval based on blur state
  useEffect(() => {
    // If question is unblurred, start checking for inactivity
    if (!isQuestionBlurred && isKeyPressed && isQuestionHovered) {
      // Check for inactivity every 100ms
      activityCheckIntervalRef.current = setInterval(checkMouseActivity, 100);
      
      // Update last activity time when unblurring
      lastActivityTimeRef.current = Date.now();
    } else {
      // Clear the interval when blurred
      if (activityCheckIntervalRef.current) {
        clearInterval(activityCheckIntervalRef.current);
        activityCheckIntervalRef.current = null;
      }
      
      // Clear any countdown
      if (countdownTimerRef.current) {
        clearTimeout(countdownTimerRef.current);
        countdownTimerRef.current = null;
        setCountdownValue(null);
      }
      
      // Clear inactivity timeout
      if (inactivityTimeoutRef.current) {
        clearTimeout(inactivityTimeoutRef.current);
        inactivityTimeoutRef.current = null;
      }
    }
    
    // Cleanup on unmount or dependency change
    return () => {
      if (activityCheckIntervalRef.current) {
        clearInterval(activityCheckIntervalRef.current);
      }
      if (inactivityTimeoutRef.current) {
        clearTimeout(inactivityTimeoutRef.current);
      }
    };
  }, [isQuestionBlurred, isKeyPressed, isQuestionHovered, checkMouseActivity]);


  
  // Update blur state based on key press and hover state
  const updateBlurState = useCallback(() => {
    // Only unblur if both conditions are met: key is pressed AND mouse is hovering
    if (isKeyPressed && isQuestionHovered) {
      // Always unblur if keys and hover conditions are met, regardless of reason
      setIsQuestionBlurred(false);
      // Reset activity timestamp when unblurring
      lastActivityTimeRef.current = Date.now();
      // Reset the blur reason
      blurReasonRef.current = 'manual';
    } else {
      // If either condition is not met, blur the question
      setIsQuestionBlurred(true);
      blurReasonRef.current = 'manual'; // This was a manual blur
      
      // Clear any countdown
      if (countdownTimerRef.current) {
        clearTimeout(countdownTimerRef.current);
        countdownTimerRef.current = null;
        setCountdownValue(null);
      }
      
      // Clear inactivity timeout
      if (inactivityTimeoutRef.current) {
        clearTimeout(inactivityTimeoutRef.current);
        inactivityTimeoutRef.current = null;
      }
    }
  }, [isKeyPressed, isQuestionHovered]);

  
  // Function to handle mouse movement
  const handleMouseMove = useCallback((e: MouseEvent) => {
    const { clientX, clientY } = e;
    
    // Update mouse position
    mousePosRef.current = { x: clientX, y: clientY };
    
    // Update last activity time
    lastActivityTimeRef.current = Date.now();
    
    // Force a re-render to update the spotlight position
    // We need to trigger a state update to re-render the component
    setMousePos({ x: clientX, y: clientY });
    
    // If the question was blurred due to inactivity and conditions are still met to unblur,
    // unblur it on mouse movement
    if (isQuestionBlurred && blurReasonRef.current === 'inactivity' && isKeyPressed && isQuestionHovered) {
      setIsQuestionBlurred(false);
      blurReasonRef.current = 'manual';
    }
    
    // If countdown is active, reset it
    if (countdownValue !== null) {
      // Clear the existing countdown
      if (countdownTimerRef.current) {
        clearTimeout(countdownTimerRef.current);
        countdownTimerRef.current = null;
      }
      setCountdownValue(null);
    }
    
    // Clear inactivity timeout if it exists
    if (inactivityTimeoutRef.current) {
      clearTimeout(inactivityTimeoutRef.current);
      inactivityTimeoutRef.current = null;
    }
  }, [countdownValue, isQuestionBlurred, isKeyPressed, isQuestionHovered]);
  
  // Add this new state at the top of your component with the other state declarations
  const [mousePos, setMousePos] = useState({ x: 0, y: 0 });

  const handleScroll = useCallback(() => {
    if (questionSectionRef.current) {
      scrollPositionRef.current = questionSectionRef.current.scrollTop;
      // Force re-render to update spotlight position when scrolling
      setMousePos(prev => ({...prev}));
    }
  }, []);
  
  // Effect to handle mouse movement in the question section
  useEffect(() => {
    // Add mouse move event listener to the question section
    const questionSection = questionSectionRef.current;
    if (questionSection && isQuestionHovered) {
      questionSection.addEventListener('mousemove', handleMouseMove);
      // Add scroll event listener
      questionSection.addEventListener('scroll', handleScroll);
      
      return () => {
        questionSection.removeEventListener('mousemove', handleMouseMove);
        questionSection.removeEventListener('scroll', handleScroll);
      };
    }
    
    return () => {};
  }, [isQuestionHovered, handleMouseMove, handleScroll]);
  
  // Effect to handle keyboard events
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Use Alt key as the designated key
      if (e.key === 'Alt') {
        e.preventDefault(); // Prevent default Alt behavior (like focusing menu)
        setIsKeyPressed(true);
      }
    };
    
    const handleKeyUp = (e: KeyboardEvent) => {
      if (e.key === 'Alt') {
        setIsKeyPressed(false);
      }
    };
    
    // Also handle blur event when window loses focus
    const handleBlur = () => {
      setIsKeyPressed(false);
    };
    
    window.addEventListener('keydown', handleKeyDown);
    window.addEventListener('keyup', handleKeyUp);
    window.addEventListener('blur', handleBlur);
    
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      window.removeEventListener('keyup', handleKeyUp);
      window.removeEventListener('blur', handleBlur);
    };
  }, []);
  
  // Update blur state when dependencies change
  useEffect(() => {
    updateBlurState();
  }, [isQuestionHovered, isKeyPressed, updateBlurState]);
  
  // Effect to clean up timers on unmount
  useEffect(() => {
    return () => {
      if (activityCheckIntervalRef.current) {
        clearInterval(activityCheckIntervalRef.current);
      }
      if (countdownTimerRef.current) {
        clearTimeout(countdownTimerRef.current);
      }
      if (inactivityTimeoutRef.current) {
        clearTimeout(inactivityTimeoutRef.current);
      }
    };
  }, []);

  
  const handleSelectQuestion = (questionId: number) => {
    setCurrentQuestionId(questionId);
    // Reset to blurred state when changing questions
    setIsQuestionBlurred(true);
    blurReasonRef.current = 'manual';
    
    // Clear any active countdown
    if (countdownTimerRef.current) {
      clearTimeout(countdownTimerRef.current);
      countdownTimerRef.current = null;
      setCountdownValue(null);
    }
    
    // Clear inactivity timeout
    if (inactivityTimeoutRef.current) {
      clearTimeout(inactivityTimeoutRef.current);
      inactivityTimeoutRef.current = null;
    }
  };
  const scrollPositionRef = useRef(0);

  const handlePrevQuestion = () => {
    const index = questions.findIndex(q => q.id === currentQuestionId);
    if (index > 0) {
      handleSelectQuestion(questions[index - 1].id);
    }
  };

  const handleNextQuestion = () => {
    const index = questions.findIndex(q => q.id === currentQuestionId);
    if (index < questions.length - 1) {
      handleSelectQuestion(questions[index + 1].id);
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
  const MCQOptions = ({ question, onSubmit }) => {
    const [selectedOption, setSelectedOption] = useState(null);
  
    const handleSubmit = () => {
      if (selectedOption !== null) {
        onSubmit(selectedOption);
      } else {
        toast({
          title: "Selection required",
          description: "Please select an option before submitting",
          variant: "destructive",
          duration: 3000
        });
      }
    };
  
    return (
      <div className="h-full flex flex-col p-6">
        <h3 className="text-lg font-medium mb-4">Select the correct answer:</h3>
        
        <div className="space-y-4 flex-1">
          {question.options?.map((option, index) => (
            <div 
              key={index}
              className={`p-4 border rounded-md cursor-pointer transition-all ${
                selectedOption === index ? 'border-blue-500 bg-blue-50' : 'border-gray-200 hover:border-gray-300'
              }`}
              onClick={() => setSelectedOption(index)}
            >
              <div className="flex items-center">
                <div className={`w-6 h-6 flex items-center justify-center rounded-full border ${
                  selectedOption === index ? 'border-blue-500 bg-blue-500' : 'border-gray-300'
                }`}>
                  {selectedOption === index && (
                    <div className="w-2 h-2 bg-white rounded-full" />
                  )}
                </div>
                <span className="ml-3">{option.text}</span>
              </div>
            </div>
          ))}
        </div>
        
        <div className="mt-auto pt-4">
          <button
            onClick={handleSubmit}
            className="w-full bg-blue-600 hover:bg-blue-700 text-white py-2 px-4 rounded transition-colors"
          >
            Submit Answer
          </button>
        </div>
      </div>
    );
  };
  // Add these functions right before your return statement in the Index component
// Update getBlurStyle to always apply blur for subjective and coding
const getBlurStyle = (): CSSProperties => {
  if (!currentQuestion) return {};
  
  const questionType = currentQuestion.type || 'coding';
  
  if (questionType === 'mcq') {
    // Complete blur/unblur for MCQ questions - unchanged
    return {
      filter: isQuestionBlurred ? 'blur(8px)' : 'none',
      transition: 'filter 0.3s ease-in-out'
    };
  } else {
    // For subjective and coding, ALWAYS apply blur regardless of Alt key state
    // This ensures the base content is always blurred
    return {
      filter: 'blur(12px)',
      transition: 'filter 0.3s ease-in-out',
      position: 'relative'
    };
  }
};


// Completely rewrite getPartialUnblurStyle to create a vertical spotlight
const getPartialUnblurStyle = (): CSSProperties => {
  if (!currentQuestion) return { display: 'none' };
  
  const questionType = currentQuestion.type || 'coding';
  
  // Only show spotlight for subjective and coding when Alt is pressed and mouse is hovering
  if (questionType === 'mcq' || !isKeyPressed || !isQuestionHovered) {
    return { display: 'none' };
  }
  
  // Get the question section's dimensions and position
  const questionRect = questionSectionRef.current?.getBoundingClientRect();
  if (!questionRect) return { display: 'none' };
  
  // Calculate mouse position relative to the question section, accounting for scroll
  const relativeY = mousePosRef.current.y - questionRect.top + scrollPositionRef.current;
  
  // Create a vertical band that follows the mouse, accounting for scroll position
  return {
    display: 'block',
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    background: 'transparent',
    maskImage: `linear-gradient(to bottom, 
                 transparent 0%, 
                 transparent ${Math.max(0, relativeY - 100)}px, 
                 black ${Math.max(0, relativeY - 100)}px, 
                 black ${relativeY + 100}px, 
                 transparent ${relativeY + 100}px, 
                 transparent 100%)`,
    WebkitMaskImage: `linear-gradient(to bottom, 
                       transparent 0%, 
                       transparent ${Math.max(0, relativeY - 100)}px, 
                       black ${Math.max(0, relativeY - 100)}px, 
                       black ${relativeY + 100}px, 
                       transparent ${relativeY + 100}px, 
                       transparent 100%)`,
    zIndex: 10,
    overflow: 'auto',
    pointerEvents: 'none',
    // Match the scroll position of the original content
    top: `-${scrollPositionRef.current}px`
  } as CSSProperties;
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
      <div 
  ref={questionSectionRef}
  className="w-1/2 relative border-r border-assessment-border overflow-y-auto transition-all duration-300 ease-in-out select-none"
  style={{
    WebkitUserSelect: 'none',
    MozUserSelect: 'none',
    msUserSelect: 'none',
    userSelect: 'none'
  }}
  onMouseEnter={() => setIsQuestionHovered(true)}
  onMouseLeave={() => {
    setIsQuestionHovered(false);
    setIsQuestionBlurred(true);
    blurReasonRef.current = 'manual';
    
    if (countdownTimerRef.current) {
      clearTimeout(countdownTimerRef.current);
      countdownTimerRef.current = null;
      setCountdownValue(null);
    }
    
    if (inactivityTimeoutRef.current) {
      clearTimeout(inactivityTimeoutRef.current);
      inactivityTimeoutRef.current = null;
    }
  }}
>
        {/* Main content - blurred when needed */}
{/* Main content - always blurred for subjective/coding */}
<div style={getBlurStyle()}>
  <QuestionDisplay
    question={currentQuestion}
    onPrevQuestion={handlePrevQuestion}
    onNextQuestion={handleNextQuestion}
    hasPrevQuestion={questions.findIndex(q => q.id === currentQuestionId) > 0}
    hasNextQuestion={questions.findIndex(q => q.id === currentQuestionId) < questions.length - 1}
  />
</div>

{/* Spotlight overlay for subjective and coding questions - not dependent on isQuestionBlurred */}
{(currentQuestion?.type === 'coding' || currentQuestion?.type === 'subjective') && isKeyPressed && isQuestionHovered && (
  <div style={getPartialUnblurStyle()}>
    <QuestionDisplay
      question={currentQuestion}
      onPrevQuestion={handlePrevQuestion}
      onNextQuestion={handleNextQuestion}
      hasPrevQuestion={questions.findIndex(q => q.id === currentQuestionId) > 0}
      hasNextQuestion={questions.findIndex(q => q.id === currentQuestionId) < questions.length - 1}
    />
  </div>
)}


          {countdownValue !== null && !isQuestionBlurred && (
            <div className="absolute top-4 right-4 bg-black/70 text-white px-3 py-2 rounded-md font-medium flex items-center z-50">
              <span>Move mouse or blur in: </span>
              <span className="ml-2 text-xl">{countdownValue}</span>
            </div>
          )}
          
          {inactivityTimeoutRef.current !== null && countdownValue === null && !isQuestionBlurred && (
            <div className="absolute top-4 right-4 bg-yellow-500/80 text-black px-3 py-2 rounded-md font-medium flex items-center z-50">
              <span>Mouse inactive! Move to prevent timer</span>
            </div>
          )}
          
          {!isQuestionBlurred && (
            <div className="absolute bottom-4 right-4 bg-yellow-500/80 text-black px-3 py-1 rounded-md text-sm font-medium z-40">
              Keep Alt pressed and move mouse to prevent blurring
            </div>
          )}
          
          {isQuestionBlurred && isQuestionHovered && (
            <div className="absolute inset-0 flex items-center justify-center bg-black/20 pointer-events-none z-40">
              <p className="text-lg font-medium text-white bg-black/60 px-4 py-2 rounded">
                Hold Alt key and move mouse to view the question
              </p>
            </div>
          )}
        </div>
        
        <div className="w-1/2 overflow-hidden">
  {currentQuestion.type === 'mcq' ? (
    <MCQOptions 
      question={currentQuestion} 
      onSubmit={(selectedOptionIndex) => {
        const isCorrect = currentQuestion.options[selectedOptionIndex].isCorrect;
        // Mark current question as completed
        const updatedQuestions = questions.map(q => 
          q.id === currentQuestionId ? { ...q, completed: true } : q
        );
        setQuestions(updatedQuestions);
        
        // Show toast based on correctness
        toast({
          title: isCorrect ? "Correct!" : "Incorrect",
          description: isCorrect 
            ? "Your answer is correct!" 
            : `The correct answer was: ${currentQuestion.options.find(opt => opt.isCorrect)?.text}`,
          variant: isCorrect ? "default" : "destructive",
          duration: 3000
        });
      }} 
    />
  ) : (
    <CodeEditor
      questionId={currentQuestionId}
      initialCode={getLanguageStarterCode(currentQuestion.type === 'subjective' ? 'javascript' : 'javascript')}
      onSubmit={handleSubmitCode}
    />
  )}
        </div>
      </div>
      {!isFullscreen && (
  <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
    <div className="bg-white p-8 rounded-lg shadow-xl max-w-md text-center">
      <h2 className="text-xl font-bold mb-4">Full Screen Required</h2>
      <p className="mb-6">Please use full screen mode to continue with the assessment.</p>
      <button
        onClick={() => goFullScreen(setIsFullscreen)}
        className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 transition-colors"
      >
        Enter Full Screen
      </button>
    </div>
  </div>)};
  </div>


  );

};

export default Index;