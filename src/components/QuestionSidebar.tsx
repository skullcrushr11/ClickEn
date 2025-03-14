
import React from 'react';
import { cn } from "@/lib/utils";
import { ChevronLeft, CheckCircle, Circle } from "lucide-react";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Button } from "@/components/ui/button";

interface Question {
  id: number;
  title: string;
  difficulty: 'easy' | 'medium' | 'hard';
  completed: boolean;
}

interface QuestionSidebarProps {
  questions: Question[];
  currentQuestionId: number;
  onSelectQuestion: (questionId: number) => void;
  collapsed: boolean;
  onToggleCollapse: () => void;
}

const QuestionSidebar: React.FC<QuestionSidebarProps> = ({
  questions,
  currentQuestionId,
  onSelectQuestion,
  collapsed,
  onToggleCollapse
}) => {
  const getDifficultyColor = (difficulty: string) => {
    switch (difficulty) {
      case 'easy':
        return 'bg-green-100 text-green-800';
      case 'medium':
        return 'bg-yellow-100 text-yellow-800';
      case 'hard':
        return 'bg-red-100 text-red-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  return (
    <div 
      className={cn(
        "h-full bg-assessment-sidebar border-r border-assessment-border transition-all duration-300 ease-in-out flex flex-col",
        collapsed ? "w-[60px]" : "w-[280px]"
      )}
    >
      <div className="p-4 border-b border-assessment-border flex items-center justify-between">
        {!collapsed && (
          <h2 className="text-lg font-semibold text-assessment-secondary">Questions</h2>
        )}
        <Button 
          variant="ghost" 
          size="sm" 
          onClick={onToggleCollapse}
          className="ml-auto p-1 h-8 w-8"
        >
          <ChevronLeft className={cn(
            "h-5 w-5 text-assessment-secondary transition-transform",
            collapsed && "rotate-180"
          )} />
        </Button>
      </div>
      
      <ScrollArea className="flex-1">
        <div className="py-2">
          {questions.map((question) => (
            <button
              key={question.id}
              onClick={() => onSelectQuestion(question.id)}
              className={cn(
                "question-sidebar-item w-full text-left px-4 py-3 flex items-start",
                currentQuestionId === question.id && "active"
              )}
            >
              {collapsed ? (
                <div className="flex flex-col items-center justify-center w-full">
                  <span className="text-sm font-medium">{question.id}</span>
                  {question.completed ? (
                    <CheckCircle className="h-4 w-4 text-assessment-success mt-1" />
                  ) : (
                    <Circle className="h-4 w-4 text-assessment-secondary mt-1" />
                  )}
                </div>
              ) : (
                <>
                  <div className="flex items-center justify-center h-6 w-6 rounded-full bg-assessment-primary/10 text-assessment-primary text-xs font-bold mr-3">
                    {question.id}
                  </div>
                  <div className="flex-1 flex flex-col">
                    <span className="text-sm font-medium text-assessment-secondary line-clamp-2">{question.title}</span>
                    <div className="flex items-center mt-1">
                      <span className={cn(
                        "text-xs px-2 py-0.5 rounded-full",
                        getDifficultyColor(question.difficulty)
                      )}>
                        {question.difficulty.charAt(0).toUpperCase() + question.difficulty.slice(1)}
                      </span>
                      {question.completed && (
                        <span className="flex items-center ml-2 text-xs text-assessment-success">
                          <CheckCircle className="h-3 w-3 mr-1" />
                          Completed
                        </span>
                      )}
                    </div>
                  </div>
                </>
              )}
            </button>
          ))}
        </div>
      </ScrollArea>
      
      <div className="p-4 border-t border-assessment-border">
        {!collapsed && (
          <div className="text-xs text-center text-muted-foreground">
            <p className="mb-1">Assessment in progress</p>
            <p>Time remaining: 01:45:32</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default QuestionSidebar;
