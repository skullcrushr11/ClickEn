from .db import connect_to_mongo
from .UserModel import UserModel
from .QuestionModel import QuestionModel
from .TestSessionModel import TestSessionModel
from .UserSubmission import UserSubmissionModel
from json import loads

DB, resDbConnectionStr = connect_to_mongo()
resDbConnection = loads(resDbConnectionStr)
if DB is None:
    print(resDbConnection)
    exit(1)

User = UserModel(DB)
Question = QuestionModel(DB)
TestSession = TestSessionModel(DB)
UserSubmission = UserSubmissionModel(DB)
