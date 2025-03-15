
import React, { useState } from 'react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import LoginForm from '@/components/auth/LoginForm';

const LoginPage: React.FC = () => {
  const [userType, setUserType] = useState<'student' | 'organizer'>('student');
  
  return (
    <div className="flex min-h-screen items-center justify-center p-4 bg-assessment-panel">
      <div className="w-full max-w-md">
        <h1 className="text-3xl font-bold text-center mb-8">AssessQuest</h1>
        
        <Tabs 
          defaultValue="student" 
          className="w-full"
          onValueChange={(value) => setUserType(value as 'student' | 'organizer')}
        >
          <TabsList className="grid grid-cols-2 w-full mb-8">
            <TabsTrigger value="student">Student</TabsTrigger>
            <TabsTrigger value="organizer">Organizer</TabsTrigger>
          </TabsList>
          
          <TabsContent value="student">
            <LoginForm userType="student" />
          </TabsContent>
          
          <TabsContent value="organizer">
            <LoginForm userType="organizer" />
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
};

export default LoginPage;
