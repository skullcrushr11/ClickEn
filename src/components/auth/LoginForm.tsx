
import React, { useState } from 'react';
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Form, FormControl, FormDescription, FormField, FormItem, FormLabel, FormMessage } from "@/components/ui/form";
import { useForm } from "react-hook-form";
import { z } from "zod";
import { zodResolver } from "@hookform/resolvers/zod";
import { useToast } from "@/hooks/use-toast";
import { useNavigate } from 'react-router-dom';
import { authenticateUser } from '@/services/api';

const loginSchema = z.object({
  email: z.string().email({ message: "Please enter a valid email address." }),
  password: z.string().min(6, { message: "Password must be at least 6 characters." }),
});

type LoginFormValues = z.infer<typeof loginSchema>;

interface LoginFormProps {
  userType: 'student' | 'organizer';
}

const LoginForm: React.FC<LoginFormProps> = ({ userType }) => {
  const { toast } = useToast();
  const navigate = useNavigate();
  const [isLoading, setIsLoading] = useState(false);
  
  const form = useForm<LoginFormValues>({
    resolver: zodResolver(loginSchema),
    defaultValues: {
      email: "",
      password: "",
    },
  });

  const onSubmit = async (data: LoginFormValues) => {
    try {
      setIsLoading(true);
      
      // In a real app with proper backend, you would make an API call here
      // For now, we're using the MongoDB service we created
      const user = await authenticateUser(data.email, data.password);
      
      // Ensure the user type matches
      if (user.userType !== userType) {
        toast({
          title: "Login failed",
          description: `This account is not registered as a ${userType}`,
          variant: "destructive",
        });
        return;
      }
      
      // Save user data to localStorage for now
      // In a production app, you would use a more secure method like HTTP-only cookies
      localStorage.setItem('userId', user.id);
      localStorage.setItem('userType', user.userType);
      localStorage.setItem('userEmail', user.email);
      localStorage.setItem('isAuthenticated', 'true');
      
      toast({
        title: "Login successful",
        description: `Logged in as ${userType}`,
      });
      
      if (userType === 'student') {
        navigate('/student-dashboard');
      } else {
        navigate('/organizer-dashboard');
      }
    } catch (error) {
      console.error('Login error:', error);
      toast({
        title: "Login failed",
        description: "Invalid credentials. Please try again.",
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="w-full max-w-md mx-auto p-6 bg-white rounded-lg shadow-md">
      <h2 className="text-2xl font-bold text-center mb-6">{userType === 'student' ? 'Student Login' : 'Organizer Login'}</h2>
      
      <Form {...form}>
        <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-6">
          <FormField
            control={form.control}
            name="email"
            render={({ field }) => (
              <FormItem>
                <FormLabel>Email</FormLabel>
                <FormControl>
                  <Input placeholder="your.email@example.com" {...field} />
                </FormControl>
                <FormMessage />
              </FormItem>
            )}
          />
          
          <FormField
            control={form.control}
            name="password"
            render={({ field }) => (
              <FormItem>
                <FormLabel>Password</FormLabel>
                <FormControl>
                  <Input type="password" placeholder="******" {...field} />
                </FormControl>
                <FormMessage />
              </FormItem>
            )}
          />
          
          <Button type="submit" className="w-full" disabled={isLoading}>
            {isLoading ? "Logging in..." : "Log in"}
          </Button>
        </form>
      </Form>
      
      <div className="mt-4 text-center text-sm">
        <p className="text-muted-foreground">
          {userType === 'student' 
            ? "Taking an assessment? Log in with your credentials above."
            : "Assessment organizer? Log in to manage assessments and candidates."}
        </p>
      </div>
      
      <div className="mt-4 p-4 bg-orange-50 border border-orange-200 rounded-md">
        <p className="text-sm text-orange-700">
          <strong>MongoDB Connection Required:</strong> Please configure your MongoDB connection string in the environment to use this feature.
        </p>
      </div>
    </div>
  );
};

export default LoginForm;
