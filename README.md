
# MongoDB Assessment Platform

[install with nvm](https://github.com/nvm-sh/nvm#installing-and-updating)

Follow these steps:

```sh
# Step 1: Clone the repository using the project's Git URL.
git clone <YOUR_GIT_URL>

# Step 2: Navigate to the project directory.
cd <YOUR_PROJECT_NAME>

# Step 3: Install the necessary dependencies.
npm i

# Step 4: Set up your MongoDB connection
# - Create a .env file in the root directory
# - Copy the contents from .env.template
# - Replace the VITE_MONGODB_URI value with your actual MongoDB connection string

# Step 5: Start the development server with auto-reloading and an instant preview.
npm run dev
```

## MongoDB Configuration

This application requires a MongoDB database connection. Before running the app:

1. Create a `.env` file in the project root
2. Add your MongoDB connection string as follows:
   ```
   VITE_MONGODB_URI=mongodb+srv://<username>:<password>@<cluster-address>/<database-name>
   ```
3. Alternatively, you can use the MongoDB configuration page in the app to set up your connection

## What technologies are used for this project?

This project is built with:

- Vite
- TypeScript
- React
- shadcn-ui
- Tailwind CSS
- MongoDB/Mongoose (for data storage)
