
import React, { useState } from 'react';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { ScrollArea } from "@/components/ui/scroll-area";
import { ExternalLink, AlertTriangle, Search } from "lucide-react";

interface CheatEvent {
  id: string;
  studentId: string;
  studentName: string;
  timestamp: Date;
  eventType: 'tab_switch' | 'copy_paste' | 'irregular_keystrokes' | 'irregular_mouse';
  description: string;
  severity: 'low' | 'medium' | 'high';
  screenshot?: string;
}

interface Student {
  id: string;
  name: string;
  email: string;
  testId: string;
  testName: string;
  suspiciousEvents: number;
  status: 'in_progress' | 'completed' | 'flagged';
}

const mockStudents: Student[] = [
  { 
    id: '1', 
    name: 'John Smith', 
    email: 'john.smith@example.com',
    testId: 'T001',
    testName: 'Algorithm Assessment',
    suspiciousEvents: 5,
    status: 'flagged'
  },
  { 
    id: '2', 
    name: 'Emily Johnson', 
    email: 'emily.j@example.com',
    testId: 'T001',
    testName: 'Algorithm Assessment',
    suspiciousEvents: 3,
    status: 'in_progress'
  },
  { 
    id: '3', 
    name: 'Michael Chang', 
    email: 'michael.c@example.com',
    testId: 'T001',
    testName: 'Algorithm Assessment',
    suspiciousEvents: 0,
    status: 'in_progress'
  },
  { 
    id: '4', 
    name: 'Sarah Williams', 
    email: 'sarah.w@example.com',
    testId: 'T001',
    testName: 'Algorithm Assessment',
    suspiciousEvents: 1,
    status: 'completed'
  },
  { 
    id: '5', 
    name: 'David Lee', 
    email: 'david.lee@example.com',
    testId: 'T002',
    testName: 'Data Structures Quiz',
    suspiciousEvents: 7,
    status: 'flagged'
  },
];

const mockCheatEvents: CheatEvent[] = [
  {
    id: 'e1',
    studentId: '1',
    studentName: 'John Smith',
    timestamp: new Date('2023-07-10T14:32:10'),
    eventType: 'tab_switch',
    description: 'Switched to another tab 3 times within 2 minutes',
    severity: 'medium'
  },
  {
    id: 'e2',
    studentId: '1',
    studentName: 'John Smith',
    timestamp: new Date('2023-07-10T14:36:22'),
    eventType: 'copy_paste',
    description: 'Attempted to paste text from clipboard',
    severity: 'high'
  },
  {
    id: 'e3',
    studentId: '1',
    studentName: 'John Smith',
    timestamp: new Date('2023-07-10T14:40:15'),
    eventType: 'irregular_keystrokes',
    description: 'Sudden burst of typing detected (120 WPM)',
    severity: 'medium'
  },
  {
    id: 'e4',
    studentId: '2',
    studentName: 'Emily Johnson',
    timestamp: new Date('2023-07-10T15:02:30'),
    eventType: 'tab_switch',
    description: 'Switched to another tab for 45 seconds',
    severity: 'low'
  },
  {
    id: 'e5',
    studentId: '2',
    studentName: 'Emily Johnson',
    timestamp: new Date('2023-07-10T15:15:47'),
    eventType: 'irregular_mouse',
    description: 'Unusual mouse movement patterns detected',
    severity: 'low'
  },
  {
    id: 'e6',
    studentId: '5',
    studentName: 'David Lee',
    timestamp: new Date('2023-07-11T10:12:05'),
    eventType: 'copy_paste',
    description: 'Multiple paste attempts detected',
    severity: 'high'
  },
  {
    id: 'e7',
    studentId: '5',
    studentName: 'David Lee',
    timestamp: new Date('2023-07-11T10:18:22'),
    eventType: 'irregular_keystrokes',
    description: 'Unusual typing rhythm and speed variations',
    severity: 'medium'
  },
];

const CheatDetectionPanel: React.FC = () => {
  const [selectedStudentId, setSelectedStudentId] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState('flagged');
  
  const filteredStudents = selectedStudentId 
    ? mockStudents.filter(student => student.id === selectedStudentId)
    : activeTab === 'all' 
      ? mockStudents 
      : activeTab === 'flagged' 
        ? mockStudents.filter(student => student.suspiciousEvents > 2) 
        : mockStudents.filter(student => student.status === activeTab);
  
  const studentEvents = selectedStudentId 
    ? mockCheatEvents.filter(event => event.studentId === selectedStudentId) 
    : [];
  
  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'low':
        return 'bg-amber-100 text-amber-800';
      case 'medium':
        return 'bg-orange-100 text-orange-800';
      case 'high':
        return 'bg-red-100 text-red-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };
  
  const getEventTypeIcon = (eventType: string) => {
    switch (eventType) {
      case 'tab_switch':
        return <div className="rounded-full bg-blue-100 p-2"><ExternalLink className="h-4 w-4 text-blue-600" /></div>;
      case 'copy_paste':
        return <div className="rounded-full bg-purple-100 p-2"><svg className="h-4 w-4 text-purple-600" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 5H6a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2v-1M8 5a2 2 0 002 2h2a2 2 0 002-2M8 5a2 2 0 012-2h2a2 2 0 012 2m0 0h2a2 2 0 012 2v3m2 4H10m0 0l3-3m-3 3l3 3" /></svg></div>;
      case 'irregular_keystrokes':
        return <div className="rounded-full bg-green-100 p-2"><svg className="h-4 w-4 text-green-600" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" /></svg></div>;
      case 'irregular_mouse':
        return <div className="rounded-full bg-pink-100 p-2"><svg className="h-4 w-4 text-pink-600" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 15l-2 5L9 9l11 4-5 2zm0 0l5 5M7.188 2.239l.777 2.897M5.136 7.965l-2.898-.777M13.95 4.05l-2.122 2.122m-5.657 5.656l-2.12 2.122" /></svg></div>;
      default:
        return <div className="rounded-full bg-gray-100 p-2"><AlertTriangle className="h-4 w-4 text-gray-600" /></div>;
    }
  };
  
  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold">Proctor Monitoring Panel</h2>
        <div className="flex items-center">
          <div className="relative">
            <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-muted-foreground" />
            <input
              className="pl-8 h-10 w-48 rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
              placeholder="Search candidates..."
            />
          </div>
        </div>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle>Candidates</CardTitle>
            <CardDescription>
              Monitor test takers and suspicious activities
            </CardDescription>
          </CardHeader>
          
          <Tabs defaultValue="flagged" value={activeTab} onValueChange={setActiveTab}>
            <CardContent className="pt-4 pb-2">
              <TabsList className="w-full grid grid-cols-4">
                <TabsTrigger value="flagged">Flagged</TabsTrigger>
                <TabsTrigger value="in_progress">In Progress</TabsTrigger>
                <TabsTrigger value="completed">Completed</TabsTrigger>
                <TabsTrigger value="all">All</TabsTrigger>
              </TabsList>
            </CardContent>
            
            <CardContent className="p-0">
              <TabsContent value="flagged" className="m-0">
                <StudentList 
                  students={filteredStudents} 
                  onSelectStudent={setSelectedStudentId}
                  selectedStudentId={selectedStudentId}
                />
              </TabsContent>
              
              <TabsContent value="in_progress" className="m-0">
                <StudentList 
                  students={filteredStudents} 
                  onSelectStudent={setSelectedStudentId}
                  selectedStudentId={selectedStudentId}
                />
              </TabsContent>
              
              <TabsContent value="completed" className="m-0">
                <StudentList 
                  students={filteredStudents} 
                  onSelectStudent={setSelectedStudentId}
                  selectedStudentId={selectedStudentId}
                />
              </TabsContent>
              
              <TabsContent value="all" className="m-0">
                <StudentList 
                  students={filteredStudents} 
                  onSelectStudent={setSelectedStudentId}
                  selectedStudentId={selectedStudentId}
                />
              </TabsContent>
            </CardContent>
          </Tabs>
        </Card>
        
        <Card>
          <CardHeader className="pb-3">
            <CardTitle>Suspicious Events</CardTitle>
            <CardDescription>
              Detailed log of detected suspicious activities
            </CardDescription>
          </CardHeader>
          
          <CardContent className="p-0">
            {selectedStudentId ? (
              studentEvents.length > 0 ? (
                <ScrollArea className="h-[420px]">
                  <div className="p-4 space-y-4">
                    {studentEvents.map(event => (
                      <div key={event.id} className="flex gap-4 p-3 bg-muted/40 rounded-lg">
                        {getEventTypeIcon(event.eventType)}
                        
                        <div className="flex-1">
                          <div className="flex items-center justify-between mb-1">
                            <span className="font-medium">{event.description}</span>
                            <Badge className={getSeverityColor(event.severity)}>
                              {event.severity.charAt(0).toUpperCase() + event.severity.slice(1)}
                            </Badge>
                          </div>
                          
                          <div className="text-sm text-muted-foreground">
                            {event.timestamp.toLocaleString()}
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </ScrollArea>
              ) : (
                <div className="px-4 py-10 text-center">
                  <p className="text-muted-foreground">No suspicious events detected for this student.</p>
                </div>
              )
            ) : (
              <div className="px-4 py-10 text-center">
                <p className="text-muted-foreground">Select a student to view their suspicious activity log.</p>
              </div>
            )}
          </CardContent>
          
          {selectedStudentId && studentEvents.length > 0 && (
            <CardFooter className="border-t p-4 flex justify-end gap-2">
              <Button variant="outline">Dismiss All</Button>
              <Button variant="destructive">Flag as Cheating</Button>
            </CardFooter>
          )}
        </Card>
      </div>
      
      {selectedStudentId && (
        <Alert variant="destructive" className="bg-destructive/10">
          <AlertTriangle className="h-4 w-4" />
          <AlertTitle>Attention Required</AlertTitle>
          <AlertDescription>
            Multiple suspicious activities detected for this candidate. Please review the logs and take appropriate action.
          </AlertDescription>
        </Alert>
      )}
    </div>
  );
};

interface StudentListProps {
  students: Student[];
  onSelectStudent: (id: string) => void;
  selectedStudentId: string | null;
}

const StudentList: React.FC<StudentListProps> = ({ students, onSelectStudent, selectedStudentId }) => {
  return (
    <ScrollArea className="h-[420px]">
      <div className="divide-y">
        {students.length > 0 ? (
          students.map(student => (
            <div 
              key={student.id} 
              className={`p-4 hover:bg-muted/60 cursor-pointer transition-colors ${selectedStudentId === student.id ? 'bg-muted' : ''}`}
              onClick={() => onSelectStudent(student.id)}
            >
              <div className="flex items-center justify-between mb-1">
                <h3 className="font-medium">{student.name}</h3>
                {student.suspiciousEvents > 2 && (
                  <Badge variant="destructive" className="text-[10px]">
                    {student.suspiciousEvents} alerts
                  </Badge>
                )}
              </div>
              
              <div className="text-sm text-muted-foreground mb-2">{student.email}</div>
              
              <div className="flex items-center text-xs">
                <span className="text-muted-foreground">{student.testName}</span>
                <span className="mx-2">•</span>
                <Badge 
                  variant={student.status === 'flagged' ? 'destructive' : 'secondary'}
                  className="text-[10px]"
                >
                  {student.status === 'in_progress' ? 'In Progress' : 
                   student.status === 'completed' ? 'Completed' : 'Flagged'}
                </Badge>
              </div>
            </div>
          ))
        ) : (
          <div className="p-6 text-center">
            <p className="text-muted-foreground">No students found in this category.</p>
          </div>
        )}
      </div>
    </ScrollArea>
  );
};

export default CheatDetectionPanel;
