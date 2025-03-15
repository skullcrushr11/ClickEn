
import React, { useState } from 'react';
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Form, FormControl, FormDescription, FormField, FormItem, FormLabel, FormMessage } from "@/components/ui/form";
import { useForm } from "react-hook-form";
import { z } from "zod";
import { zodResolver } from "@hookform/resolvers/zod";
import { useToast } from "@/hooks/use-toast";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { connectToDatabase } from '@/config/mongodb';

const dbConfigSchema = z.object({
  mongodbUri: z.string().min(10, { message: "Please enter a valid MongoDB URI." }),
});

type DBConfigFormValues = z.infer<typeof dbConfigSchema>;

const MongoDBConfig: React.FC = () => {
  const { toast } = useToast();
  const [isConnecting, setIsConnecting] = useState(false);
  const [isConnected, setIsConnected] = useState(false);
  
  const form = useForm<DBConfigFormValues>({
    resolver: zodResolver(dbConfigSchema),
    defaultValues: {
      mongodbUri: "",
    },
  });

  const onSubmit = async (data: DBConfigFormValues) => {
    try {
      setIsConnecting(true);
      
      // Store MongoDB URI in localStorage (for demo purposes only)
      // In a production app, you would use a more secure method
      localStorage.setItem('MONGODB_URI', data.mongodbUri);
      
      // Set the environment variable (this is a mock implementation)
      // In a real environment, this would be set server-side
      (process.env as any).MONGODB_URI = data.mongodbUri;
      
      // Test the connection
      await connectToDatabase();
      
      setIsConnected(true);
      
      toast({
        title: "Connection successful",
        description: "Successfully connected to MongoDB",
      });
    } catch (error) {
      console.error('MongoDB connection error:', error);
      setIsConnected(false);
      
      toast({
        title: "Connection failed",
        description: "Failed to connect to MongoDB. Please check your URI and try again.",
        variant: "destructive",
      });
    } finally {
      setIsConnecting(false);
    }
  };

  return (
    <Card className="w-full max-w-md mx-auto">
      <CardHeader>
        <CardTitle>MongoDB Configuration</CardTitle>
        <CardDescription>
          Configure your MongoDB connection for storing application data.
        </CardDescription>
      </CardHeader>
      
      <CardContent>
        <Form {...form}>
          <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-6">
            <FormField
              control={form.control}
              name="mongodbUri"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>MongoDB Connection URI</FormLabel>
                  <FormControl>
                    <Input 
                      placeholder="mongodb+srv://username:password@cluster.mongodb.net/database"
                      type="password" 
                      {...field} 
                    />
                  </FormControl>
                  <FormDescription>
                    Your MongoDB connection string including username, password, and database name.
                  </FormDescription>
                  <FormMessage />
                </FormItem>
              )}
            />
            
            <Button type="submit" className="w-full" disabled={isConnecting}>
              {isConnecting ? "Connecting..." : isConnected ? "Connected" : "Connect to MongoDB"}
            </Button>
          </form>
        </Form>
      </CardContent>
      
      <CardFooter className="flex flex-col text-sm text-muted-foreground">
        <p className="mb-2">
          Format: <code>mongodb+srv://username:password@cluster.mongodb.net/database</code>
        </p>
        <p>
          Note: In a production environment, connection strings should be stored securely as environment variables.
        </p>
      </CardFooter>
    </Card>
  );
};

export default MongoDBConfig;
