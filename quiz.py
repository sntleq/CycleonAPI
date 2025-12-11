from typing import Optional
from db import get_db
from datetime import datetime
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()


# Pydantic models for request bodies
class UserCreate(BaseModel):
    username: str
    name: str
    password: str


class QuizCreate(BaseModel):
    user_id: int
    score: int
    insights: Optional[str] = None


# Initialize tables
@router.post("/init-tables")
async def init_tables():
    """Create user and quiz tables if they don't exist."""
    with get_db() as conn:
        cursor = conn.cursor()

        # Create users table
        cursor.execute("""
                       CREATE TABLE IF NOT EXISTS users
                       (
                           id         SERIAL PRIMARY KEY,
                           username   VARCHAR(255) UNIQUE NOT NULL,
                           name       VARCHAR(255)        NOT NULL,
                           password   VARCHAR(255)        NOT NULL,
                           created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                       )
                       """)

        # Create quiz table
        cursor.execute("""
                       CREATE TABLE IF NOT EXISTS quiz
                       (
                           id         SERIAL PRIMARY KEY,
                           user_id    INTEGER NOT NULL,
                           score      INTEGER NOT NULL,
                           insights   TEXT,
                           created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                           FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
                       )
                       """)

        conn.commit()
        cursor.close()

    return {"message": "Tables created successfully"}


# User endpoints
@router.post("/users")
async def create_user(user: UserCreate):
    """Create a new user."""
    with get_db() as conn:
        cursor = conn.cursor()

        try:
            cursor.execute("""
                           INSERT INTO users (username, name, password)
                           VALUES (%s, %s, %s)
                           RETURNING id, username, name, created_at
                           """, (user.username, user.name, user.password))

            result = cursor.fetchone()
            conn.commit()
            cursor.close()

            return {
                "id": result["id"],
                "username": result["username"],
                "name": result["name"],
                "created_at": result["created_at"]
            }
        except Exception as e:
            conn.rollback()
            cursor.close()
            if "unique" in str(e).lower():
                raise HTTPException(status_code=400, detail="Username already exists")
            raise HTTPException(status_code=500, detail=str(e))


@router.get("/users")
async def list_users():
    """Get all users."""
    with get_db() as conn:
        cursor = conn.cursor()

        cursor.execute("""
                       SELECT id, username, name, created_at
                       FROM users
                       ORDER BY created_at DESC
                       """)

        users = cursor.fetchall()
        cursor.close()

    return [dict(row) for row in users]


@router.get("/users/{user_id}")
async def get_user(user_id: int):
    """Get a specific user by ID."""
    with get_db() as conn:
        cursor = conn.cursor()

        cursor.execute("""
                       SELECT id, username, name, created_at
                       FROM users
                       WHERE id = %s
                       """, (user_id,))

        user = cursor.fetchone()
        cursor.close()

    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    return dict(user)


# Quiz endpoints
@router.post("/quiz")
async def create_quiz(quiz: QuizCreate):
    """Create a new quiz entry."""
    with get_db() as conn:
        cursor = conn.cursor()

        # Check if user exists
        cursor.execute("SELECT id FROM users WHERE id = %s", (quiz.user_id,))
        if not cursor.fetchone():
            cursor.close()
            raise HTTPException(status_code=404, detail="User not found")

        try:
            cursor.execute("""
                           INSERT INTO quiz (user_id, score, insights)
                           VALUES (%s, %s, %s)
                           RETURNING id, user_id, score, insights, created_at
                           """, (quiz.user_id, quiz.score, quiz.insights))

            result = cursor.fetchone()
            conn.commit()
            cursor.close()

            return dict(result)
        except Exception as e:
            conn.rollback()
            cursor.close()
            raise HTTPException(status_code=500, detail=str(e))


@router.get("/quiz")
async def list_quizzes():
    """Get all quiz entries with user information."""
    with get_db() as conn:
        cursor = conn.cursor()

        cursor.execute("""
                       SELECT q.id,
                              q.user_id,
                              u.username,
                              u.name,
                              q.score,
                              q.insights,
                              q.created_at
                       FROM quiz q
                                JOIN users u ON q.user_id = u.id
                       ORDER BY q.created_at DESC
                       """)

        quizzes = cursor.fetchall()
        cursor.close()

    return [dict(row) for row in quizzes]


@router.get("/quiz/user/{user_id}")
async def get_user_quizzes(user_id: int):
    """Get all quiz entries for a specific user."""
    with get_db() as conn:
        cursor = conn.cursor()

        cursor.execute("""
                       SELECT q.id,
                              q.user_id,
                              q.score,
                              q.insights,
                              q.created_at
                       FROM quiz q
                       WHERE q.user_id = %s
                       ORDER BY q.created_at DESC
                       """, (user_id,))

        quizzes = cursor.fetchall()
        cursor.close()

    if not quizzes:
        # Check if user exists
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM users WHERE id = %s", (user_id,))
            if not cursor.fetchone():
                cursor.close()
                raise HTTPException(status_code=404, detail="User not found")
            cursor.close()

    return [dict(row) for row in quizzes]


@router.get("/quiz/{quiz_id}")
async def get_quiz(quiz_id: int):
    """Get a specific quiz entry by ID."""
    with get_db() as conn:
        cursor = conn.cursor()

        cursor.execute("""
                       SELECT q.id,
                              q.user_id,
                              u.username,
                              u.name,
                              q.score,
                              q.insights,
                              q.created_at
                       FROM quiz q
                                JOIN users u ON q.user_id = u.id
                       WHERE q.id = %s
                       """, (quiz_id,))

        quiz = cursor.fetchone()
        cursor.close()

    if not quiz:
        raise HTTPException(status_code=404, detail="Quiz not found")

    return dict(quiz)