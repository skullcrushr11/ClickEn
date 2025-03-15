
import React from 'react';
import { cn } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { ScrollArea } from "@/components/ui/scroll-area";
import { ArrowLeft, ArrowRight, Clock, Tag, BarChart2 } from "lucide-react";
import { Button } from "@/components/ui/button";

interface QuestionDisplayProps {
  question: {
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
    timeComplexity?: string;
    spaceComplexity?: string;
  };
  onPrevQuestion: () => void;
  onNextQuestion: () => void;
  hasNextQuestion: boolean;
  hasPrevQuestion: boolean;
}

const QuestionDisplay: React.FC<QuestionDisplayProps> = ({
  question,
  onPrevQuestion,
  onNextQuestion,
  hasNextQuestion,
  hasPrevQuestion
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
    <div className="h-full flex flex-col bg-assessment-panel animate-fade-in">
      <div className="px-6 py-4 border-b border-assessment-border flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <span className="flex items-center justify-center h-8 w-8 rounded-full bg-assessment-primary/10 text-assessment-primary font-semibold">
            {question.id}
          </span>
          <h1 className="text-xl font-semibold text-assessment-secondary">{question.title}</h1>
          <Badge className={cn(
            "ml-2",
            getDifficultyColor(question.difficulty)
          )}>
            {question.difficulty.charAt(0).toUpperCase() + question.difficulty.slice(1)}
          </Badge>
        </div>
        <div className="flex items-center space-x-2">
          <Button
            variant="outline"
            size="sm"
            onClick={onPrevQuestion}
            disabled={!hasPrevQuestion}
            className="flex items-center"
          >
            <ArrowLeft className="mr-1 h-4 w-4" />
            Prev
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={onNextQuestion}
            disabled={!hasNextQuestion}
            className="flex items-center"
          >
            Next
            <ArrowRight className="ml-1 h-4 w-4" />
          </Button>
        </div>
      </div>

      <ScrollArea className="flex-1 p-6">
        <div className="max-w-3xl mx-auto question-content">
          <div className="prose prose-sm max-w-none">
            <p className="text-assessment-secondary whitespace-pre-line">
              {question.description}
            </p>

            <Separator className="my-4" />

            <h3 className="text-lg font-medium mt-6 mb-3">Examples</h3>
            {question.examples.map((example, idx) => (
              <div key={idx} className="mb-4 bg-assessment-sidebar rounded-md p-4">
                <p className="font-medium text-sm mb-1">Example {idx + 1}:</p>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <p className="text-xs font-medium text-assessment-secondary mb-1">Input:</p>
                    <pre className="text-sm p-2 bg-white rounded border border-assessment-border">
                      {example.input}
                    </pre>
                  </div>
                  <div>
                    <p className="text-xs font-medium text-assessment-secondary mb-1">Output:</p>
                    <pre className="text-sm p-2 bg-white rounded border border-assessment-border">
                      {example.output}
                    </pre>
                  </div>
                </div>
                {example.explanation && (
                  <div className="mt-2">
                    <p className="text-xs font-medium text-assessment-secondary mb-1">Explanation:</p>
                    <p className="text-sm text-muted-foreground">{example.explanation}</p>
                  </div>
                )}
              </div>
            ))}

            <h3 className="text-lg font-medium mt-6 mb-3">Constraints</h3>
            <ul className="list-disc pl-5 space-y-1">
              {question.constraints.map((constraint, idx) => (
                <li key={idx} className="text-sm text-assessment-secondary">
                  {constraint}
                </li>
              ))}
            </ul>

            <Separator className="my-6" />

            <div className="grid grid-cols-3 gap-4 mt-4">
              <div className="flex items-center">
                <Clock className="h-4 w-4 text-muted-foreground mr-2" />
                <span className="text-xs text-muted-foreground">
                  {question.timeComplexity || "Expected Time: O(n)"}
                </span>
              </div>
              <div className="flex items-center">
                <BarChart2 className="h-4 w-4 text-muted-foreground mr-2" />
                <span className="text-xs text-muted-foreground">
                  {question.spaceComplexity || "Expected Space: O(1)"}
                </span>
              </div>
              <div className="flex items-center">
                <Tag className="h-4 w-4 text-muted-foreground mr-2" />
                <div className="flex flex-wrap gap-1">
                  {question.tags.map((tag, idx) => (
                    <span key={idx} className="text-xs bg-muted text-muted-foreground px-2 py-0.5 rounded">
                      {tag}
                    </span>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      </ScrollArea>
    </div>
  );
};

export default QuestionDisplay;
