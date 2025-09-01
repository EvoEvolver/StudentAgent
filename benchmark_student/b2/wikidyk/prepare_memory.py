import os
import json
import logging
import pandas as pd
from typing import List, Dict
from datetime import datetime
from student.agent import *
from student.agent.memory_rag import MemoryRAG, MemoryNodeRAG
from student.agent.agent_baselines import BaselineAgent
import dotenv
dotenv.load_dotenv()

def setup_logger(training_run_id: str) -> logging.Logger:
    """
    Configure logging for the memory preparation process.
    
    Args:
        training_run_id: ID of the current training run
    
    Returns:
        Configured logger instance
    """
    # Create logs directory if it doesn't exist
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # Set up logger
    logger = logging.getLogger(f"memory_prep_{training_run_id}")
    logger.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler
    log_file = os.path.join(log_dir, f"prepare_memory_{training_run_id}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    # Stream handler for console output
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def load_data(n: int) -> pd.DataFrame:
    """
    Load WikiDYK dataset from Hugging Face and return n samples.
    
    Args:
        n: Number of samples to load
        
    Returns:
        DataFrame with fact and eval columns
    
    Raises:
        Exception: If dataset cannot be loaded
    """
    try:
        wikidyk = pd.read_parquet("hf://datasets/YWZBrandon/wikidyk/data/test-00000-of-00001.parquet")
        wikidyk_data = wikidyk[["fact", "eval"]].drop_duplicates()
        return wikidyk_data.head(n)
    except Exception as e:
        raise Exception(f"Failed to load WikiDYK dataset: {str(e)}")


def make_rag_memory(facts: List[str], RAG_MEMORY_PATH: str, logger: logging.Logger) -> None:
    """
    Create and save a RAG memory from facts.
    
    Args:
        facts: List of facts to add to memory
        RAG_MEMORY_PATH: Path to save the RAG memory
        logger: Logger instance for tracking progress
        
    Raises:
        ValueError: If embeddings are inconsistent
    """
    os.makedirs(os.path.dirname(RAG_MEMORY_PATH), exist_ok=True)
    logger.info(f"Creating RAG memory with {len(facts)} facts")
    
    memory_rag_naive = MemoryRAG()
    for i, fact in enumerate(facts):
        try:
            new_node = MemoryNodeRAG(input=fact)
            memory_rag_naive.add(new_node)
            if (i + 1) % 10 == 0:
                logger.info(f"Added {i+1}/{len(facts)} facts to RAG memory")
        except Exception as e:
            logger.warning(f"Failed to add fact to RAG memory: {fact}")
            logger.error(f"Error: {str(e)}")
    
    memory_rag_naive.get_nodes()
    embedding_lengths = {len(node.embeddings) for node in memory_rag_naive.memory.values()}
    if len(embedding_lengths) != 1 or 0 in embedding_lengths:
        logger.error("Inconsistent embedding lengths detected in RAG memory")
        raise ValueError("Inconsistent embedding lengths detected in RAG memory")

    try:
        memory_rag_naive.save(RAG_MEMORY_PATH)
        logger.info(f"RAG memory saved to {RAG_MEMORY_PATH}")
    except Exception as e:
        logger.error(f"Failed to save RAG memory: {str(e)}")
        raise Exception(f"Failed to save RAG memory: {str(e)}")


def make_student_memory(facts: List[str], AGENT_CONFIG: dict, TRAINING_STUDENT_MEMORY_PATH: str, logger: logging.Logger) -> StudentAgent:
    """
    Create and train a StudentAgent with given facts.
    
    Args:
        facts: List of facts to train on
        AGENT_CONFIG: Configuration for StudentAgent
        TRAINING_STUDENT_MEMORY_PATH: Path to save training checkpoints
        logger: Logger instance for tracking progress
        
    Returns:
        Trained StudentAgent instance
    """
    # Create checkpoint directory if it doesn't exist
    os.makedirs(os.path.dirname(TRAINING_STUDENT_MEMORY_PATH), exist_ok=True)
    
    logger.info("Initializing StudentAgent for training")
    teaching_prompt = "You are an expert in collecting factual knowledge in your memory. Memorize the facts explicitly. (NO verification required)"
    student_wiki = StudentAgent(**AGENT_CONFIG)
    student_wiki.reset_system_prompt(teaching_prompt, append=True)
    student_wiki.save(TRAINING_STUDENT_MEMORY_PATH)
    logger.info(f"StudentAgent initialized and saved to {TRAINING_STUDENT_MEMORY_PATH}")

    def train_memory(fact: str):
        try:
            student_wiki.load(TRAINING_STUDENT_MEMORY_PATH)
            student_wiki.reset_chat()
            p = f"Fact: {fact}"
            student_wiki.run(p, remove_tools=["ask memory"])
            student_wiki.reset_chat()
            student_wiki.save(TRAINING_STUDENT_MEMORY_PATH)
        except Exception as e:
            logger.warning(f"Failed to train on fact: {fact}")
            logger.error(f"Error: {str(e)}")

    logger.info(f"Starting training on {len(facts)} facts")
    for j, fact in enumerate(facts):
        logger.info(f"Training on fact {j+1}/{len(facts)}")
        train_memory(fact)
    
    logger.info("Student memory training completed")
    return student_wiki


def safe_student_memory(student_wiki: StudentAgent, TRAINING_STUDENT_MEMORY_PATH: str, STUDENT_MEMORY_PATH: str, logger: logging.Logger) -> None:
    """
    Create a clean copy of the trained student memory.
    
    Args:
        student_wiki: Trained StudentAgent instance
        TRAINING_STUDENT_MEMORY_PATH: Path to training checkpoint
        STUDENT_MEMORY_PATH: Path to save clean memory
        logger: Logger instance for tracking progress
    """
    os.makedirs(os.path.dirname(STUDENT_MEMORY_PATH), exist_ok=True)
    logger.info("Creating clean memory copy")
    
    try:
        student_wiki.load(TRAINING_STUDENT_MEMORY_PATH)
        student_wiki.reset_conversation()
        student_wiki.save(STUDENT_MEMORY_PATH)
        logger.info(f"Clean memory saved to {STUDENT_MEMORY_PATH}")
    except Exception as e:
        error_msg = f"Failed to save clean memory: {str(e)}"
        logger.error(error_msg)
        raise Exception(error_msg)


def main():
    # Configuration
    training_run_id = "run__100"
    
    # Set up logging
    logger = setup_logger(training_run_id)
    logger.info(f"Starting memory preparation for training run {training_run_id}")
    
    n_wikidyk = 100

    # Paths
    RAG_MEMORY_PATH = f"memory/wikidyk_rag__{training_run_id}.parquet"
    TRAINING_STUDENT_MEMORY_PATH = f"checkpoints/training_{training_run_id}"
    STUDENT_MEMORY_PATH = f"checkpoints/memory_{training_run_id}"

    # Agent configuration
    AGENT_CONFIG = {
        "expensive": False,
        "provider": "anthropic",
        "cache": False
    }
    logger.info(f"Agent configuration: {AGENT_CONFIG}")

    try:
        # Load and prepare data
        logger.info(f"Loading {n_wikidyk} samples from WikiDYK dataset")
        data = load_data(n_wikidyk)
        facts = list(data["fact"])
        logger.info(f"Loaded {len(facts)} facts")

        # Create memories
        make_rag_memory(facts, RAG_MEMORY_PATH, logger)
        
        student_agent = make_student_memory(facts, AGENT_CONFIG, TRAINING_STUDENT_MEMORY_PATH, logger)
        
        safe_student_memory(student_agent, TRAINING_STUDENT_MEMORY_PATH, STUDENT_MEMORY_PATH, logger)
        
        logger.info("Memory preparation completed successfully")
        
    except Exception as e:
        logger.error(f"Error in memory preparation: {str(e)}")
        raise
    finally:
        # Add separator for next run
        logger.info("-" * 80)


if __name__ == "__main__":
    main()