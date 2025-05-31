"""
Documentation Analysis System - Agent 1

This module implements a documentation analyzer that evaluates technical documentation
for readability, structure, completeness, and style guidelines. It uses a combination
of rule-based analysis and LLM assessment to provide comprehensive feedback.

Key Features:
- Content extraction using Selenium
- Readability analysis using textstat
- Style analysis based on Microsoft Style Guide
- Comprehensive reporting with actionable suggestions
"""

import os
import json
import time
import re
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import google.generativeai as genai
import textstat
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()

@dataclass
class StyleIssue:
    """Represents a style guideline violation with severity and suggestions."""
    issue_type: str
    location: str
    current_text: str
    suggestion: str
    severity: str  # 'high', 'medium', 'low'

class DocumentationAnalyzer:
    """Main class for documentation analysis functionality."""
    
    def __init__(self) -> None:
        """Initialize the analyzer with configuration and settings."""
        self.configure_genai()
        self.setup_constants()
        self.response_cache = {}  # Cache for API responses
    
    def setup_constants(self) -> None:
        """Set up constant values used throughout the analysis."""
        self.SELECTORS = [
            "div.article__body.markdown",
            "div.article-body",
            "article",
            "div.article-content"
        ]
        self.WAIT_TIMEOUT = 20
        self.MAX_SENTENCE_LENGTH = 25
        self.MAX_PARAGRAPH_LENGTH = 100
        self.VALID_URL_PATTERN = re.compile(
            r'^https?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
            r'localhost|'  # localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    
    def configure_genai(self) -> bool:
        """Configure the Gemini API with the provided key.
        
        Returns:
            bool: True if configuration successful, False otherwise
        """
        try:
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY not found in environment variables")
            
            print("\nConfiguring Gemini API...")
            genai.configure(api_key=api_key)
            print("Gemini API configured successfully")
            return True
        except Exception as e:
            print(f"Error configuring Gemini API: {e}")
            return False

    def setup_driver(self) -> webdriver.Chrome:
        """Set up and configure the Chrome WebDriver.
        
        Returns:
            webdriver.Chrome: Configured Chrome WebDriver instance
        """
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--window-size=1920,1080')
        chrome_options.add_argument(
            '--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
            'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        )
        
        service = Service(ChromeDriverManager().install())
        return webdriver.Chrome(service=service, options=chrome_options)

    def validate_url(self, url: str) -> bool:
        """Validate if the provided URL is properly formatted.
        
        Args:
            url: The URL to validate
            
        Returns:
            bool: True if URL is valid, False otherwise
        """
        if not url or not isinstance(url, str):
            return False
        return bool(self.VALID_URL_PATTERN.match(url))

    def fetch_content(self, url: str) -> str:
        """Fetch content from the specified URL using Selenium.
        
        Args:
            url: The URL to fetch content from
            
        Returns:
            str: The extracted content or error message
        """
        if not self.validate_url(url):
            error_msg = f"Invalid URL format: {url}"
            print(error_msg)
            return f"Error: {error_msg}"

        driver = None
        try:
            print(f"Setting up Chrome driver for URL: {url}")
            driver = self.setup_driver()
            
            print("Attempting to fetch content...")
            driver.get(url)
            
            print("Waiting for content to load...")
            wait = WebDriverWait(driver, self.WAIT_TIMEOUT)
            
            content = None
            for selector in self.SELECTORS:
                try:
                    print(f"Trying selector: {selector}")
                    element = wait.until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, selector))
                    )
                    content = element.text
                    if content and len(content.strip()) > 50:  # Ensure meaningful content
                        print(f"Found content using selector: {selector}")
                        break
                except Exception as e:
                    print(f"Selector {selector} failed: {str(e)}")
                    continue
            
            if not content or len(content.strip()) <= 50:
                print("\nWarning: Could not find meaningful article content.")
                print("Page source preview:")
                print(driver.page_source[:1000])
                if driver:
                    driver.quit()
                return "Error: Could not find meaningful article content"
            
            if driver:
                driver.quit()
            return content
            
        except Exception as e:
            error_msg = f"Error fetching content: {str(e)}"
            print(error_msg)
            if driver:
                try:
                    driver.quit()
                except:
                    pass
            return f"Error: {error_msg}"

    def call_llm(self, prompt: str) -> str:
        """Call the Gemini LLM with the provided prompt.
        
        Args:
            prompt: The prompt to send to the LLM
            
        Returns:
            str: The LLM's response or error message
        """
        try:
            # Check cache first
            cache_key = hash(prompt)
            if cache_key in self.response_cache:
                print("Using cached response")
                return self.response_cache[cache_key]
            
            if not self.configure_genai():
                print("Failed to configure Gemini API")
                return "Error: Failed to configure Gemini API"
            
            print("\nSending request to Gemini API...")
            print(f"Prompt length: {len(prompt)} characters")
            
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content(prompt)
            
            # Cache the response
            self.response_cache[cache_key] = response.text
            
            print("\nReceived response from Gemini API")
            print(f"Response length: {len(response.text)} characters")
            print("First 200 characters of response:")
            print(response.text[:200])
            
            return response.text
        except Exception as e:
            print(f"Error calling LLM: {e}")
            return "Error: LLM call failed."

    def analyze_readability(self, content: str) -> Dict[str, Any]:
        """Comprehensive readability analysis using LLM assessment."""
        try:
            # Quick content preprocessing
            clean_content = self._quick_clean(content)
            word_count = len(clean_content.split())
            
            # Calculate readability scores
            flesch_score = textstat.flesch_reading_ease(clean_content)
            gunning_fog = textstat.gunning_fog(clean_content)
            
            # Get detailed readability interpretations
            flesch_interpretation = self._interpret_flesch_score(flesch_score)
            gunning_interpretation = self._interpret_gunning_fog(gunning_fog)
            
            # Analyze complex sentences
            sentences = clean_content.split('.')
            complex_sentences = [
                s for s in sentences 
                if len(s.split()) > self.MAX_SENTENCE_LENGTH
            ]
            
            # Skip analysis if content is too short
            if word_count < 50:
                return {
                    "status": "too_short",
                    "word_count": word_count,
                    "flesch_score": flesch_interpretation,
                    "gunning_fog": gunning_interpretation,
                    "complex_sentences": len(complex_sentences),
                    "llm_assessment": "Content too short for meaningful analysis"
                }
            
            # Handle very long content
            if word_count > 3000:
                clean_content = ' '.join(clean_content.split()[:3000])
            
            # Focused LLM prompt for comprehensive assessment
            prompt = self._generate_readability_prompt(clean_content, flesch_score, gunning_fog)
            llm_assessment = self.call_llm(prompt)
            
            # Parse assessment scores
            scores = self._parse_assessment_scores(llm_assessment)
            
            return {
                "word_count": word_count,
                "flesch_score": flesch_interpretation,
                "gunning_fog": gunning_interpretation,
                "complex_sentences": len(complex_sentences),
                "llm_assessment": llm_assessment,
                "assessment_scores": scores,
                "quick_verdict": self._generate_verdict(scores),
                "industry_targets": {
                    "general_public": {"flesch": "60-70", "fog": "7-9"},
                    "business": {"flesch": "50-60", "fog": "8-10"},
                    "technical": {"flesch": "40-50", "fog": "10-12"},
                    "academic": {"flesch": "30-40", "fog": "12-15"},
                    "legal_medical": {"flesch": "0-30", "fog": "15+"},
                    "moengage_target": {"flesch": "60-70", "fog": "8-10"}
                }
            }
            
        except Exception as e:
            print(f"Error in readability analysis: {e}")
            return {
                "error": str(e),
                "status": "analysis_failed",
                "flesch_score": {
                    "score": 0,
                    "description": "Error calculating score",
                    "needs_improvement": True
                },
                "gunning_fog": {
                    "score": 0,
                    "description": "Error calculating score",
                    "needs_improvement": True
                },
                "complex_sentences": 0,
                "llm_assessment": f"Error: {str(e)}"
            }

    def _interpret_flesch_score(self, score: float) -> Dict[str, Any]:
        """Interpret Flesch Reading Ease score with detailed analysis.
        
        Args:
            score: The Flesch score to interpret
            
        Returns:
            Dict containing detailed interpretation
        """
        interpretation = {
            "score": score,
            "grade_level": "",
            "description": "",
            "suitable_for": [],
            "writing_style": "",
            "target_audience": "",
            "needs_improvement": False
        }
        
        if score >= 90:
            interpretation.update({
                "grade_level": "5th grade",
                "description": "Very Easy",
                "suitable_for": ["Children", "General public", "Basic instructions"],
                "writing_style": "Simple, conversational language with short sentences",
                "target_audience": "General audience"
            })
        elif score >= 80:
            interpretation.update({
                "grade_level": "6th grade",
                "description": "Easy",
                "suitable_for": ["Basic instructions", "Popular content", "General public"],
                "writing_style": "Clear, straightforward writing",
                "target_audience": "General audience"
            })
        elif score >= 70:
            interpretation.update({
                "grade_level": "7th grade",
                "description": "Fairly Easy",
                "suitable_for": ["Consumer-facing content", "General public", "Basic technical content"],
                "writing_style": "Accessible to most readers",
                "target_audience": "General audience"
            })
        elif score >= 60:
            interpretation.update({
                "grade_level": "8th-9th grade",
                "description": "Standard",
                "suitable_for": ["Business communication", "Technical documentation", "Marketing content"],
                "writing_style": "Target zone for business communication",
                "target_audience": "Average adults"
            })
        elif score >= 50:
            interpretation.update({
                "grade_level": "10th-12th grade",
                "description": "Fairly Difficult",
                "suitable_for": ["Technical documentation", "Business reports", "Academic content"],
                "writing_style": "High school level complexity",
                "target_audience": "Educated adults"
            })
        elif score >= 30:
            interpretation.update({
                "grade_level": "College",
                "description": "Difficult",
                "suitable_for": ["Academic writing", "Technical documentation", "Professional content"],
                "writing_style": "Academic or technical writing",
                "target_audience": "College-educated readers"
            })
        else:
            interpretation.update({
                "grade_level": "Graduate",
                "description": "Very Difficult",
                "suitable_for": ["Research papers", "Legal documents", "Specialized content"],
                "writing_style": "Highly complex, specialized content",
                "target_audience": "Specialized professionals"
            })
        
        # Check if score needs improvement for MoEngage docs
        interpretation["needs_improvement"] = score < 60  # Target: 60-70
        
        return interpretation

    def _interpret_gunning_fog(self, score: float) -> Dict[str, Any]:
        """Interpret Gunning Fog Index with detailed analysis.
        
        Args:
            score: The Gunning Fog score to interpret
            
        Returns:
            Dict containing detailed interpretation
        """
        interpretation = {
            "score": score,
            "grade_level": "",
            "description": "",
            "suitable_for": [],
            "writing_style": "",
            "target_audience": "",
            "needs_improvement": False
        }
        
        if score < 7:
            interpretation.update({
                "grade_level": "Elementary school",
                "description": "Very Easy",
                "suitable_for": ["Children's content", "Basic instructions", "General public"],
                "writing_style": "Simple, clear language",
                "target_audience": "General audience"
            })
        elif score <= 8:
            interpretation.update({
                "grade_level": "Middle school",
                "description": "Ideal for most audiences",
                "suitable_for": ["General public", "Business communication", "Marketing content"],
                "writing_style": "Clear and accessible",
                "target_audience": "General audience"
            })
        elif score <= 10:
            interpretation.update({
                "grade_level": "High school freshman/sophomore",
                "description": "Good",
                "suitable_for": ["Business communication", "Technical documentation", "Marketing content"],
                "writing_style": "Balanced technical and accessible language",
                "target_audience": "Educated adults"
            })
        elif score <= 12:
            interpretation.update({
                "grade_level": "High school junior/senior",
                "description": "Acceptable",
                "suitable_for": ["Technical documentation", "Business reports", "Professional content"],
                "writing_style": "More technical but still accessible",
                "target_audience": "Educated adults"
            })
        elif score <= 16:
            interpretation.update({
                "grade_level": "College",
                "description": "Getting difficult",
                "suitable_for": ["Academic writing", "Technical documentation", "Professional content"],
                "writing_style": "Technical and specialized",
                "target_audience": "College-educated readers"
            })
        elif score <= 20:
            interpretation.update({
                "grade_level": "Graduate",
                "description": "Very difficult",
                "suitable_for": ["Research papers", "Technical documentation", "Specialized content"],
                "writing_style": "Highly technical and complex",
                "target_audience": "Specialized professionals"
            })
        else:
            interpretation.update({
                "grade_level": "Post-graduate",
                "description": "Extremely difficult",
                "suitable_for": ["Research papers", "Legal documents", "Specialized content"],
                "writing_style": "Highly specialized and complex",
                "target_audience": "Experts in the field"
            })
        
        # Check if score needs improvement for MoEngage docs
        interpretation["needs_improvement"] = score > 10  # Target: 8-10
        
        return interpretation

    def _generate_readability_prompt(self, content: str, flesch_score: float, 
                                   gunning_fog: float) -> str:
        """Generate the readability assessment prompt.
        
        Args:
            content: The content to analyze
            flesch_score: The Flesch Reading Ease score
            gunning_fog: The Gunning Fog Index score
            
        Returns:
            str: The formatted prompt
        """
        return f"""You are an expert content analyst specializing in customer engagement 
        platform documentation. Your task is to evaluate MoEngage help articles for quality 
        and usability.

        CURRENT READABILITY METRICS:
        - Flesch Reading Ease Score: {flesch_score} ({self._interpret_flesch_score(flesch_score)['description']})
        - Gunning Fog Index: {gunning_fog} ({self._interpret_gunning_fog(gunning_fog)['description']})
        
        TARGET METRICS FOR MOENGAGE DOCS:
        - Flesch Score: 60-70 (Standard level)
        - Gunning Fog: 8-10 (High school to early college level)

        TARGET AUDIENCE PERSONA:
        You're assessing content for a Growth Marketer at a consumer brand who:
        - Manages omnichannel customer engagement campaigns using MoEngage
        - Has marketing automation knowledge but limited technical depth
        - Works under tight deadlines with multiple campaigns running simultaneously
        - Needs actionable, implementable guidance to drive engagement and retention
        - Often references documentation while actively configuring campaigns
        - Values business outcomes over technical specifications

        EVALUATION FRAMEWORK:

        1. READABILITY ASSESSMENT:
        - Language accessibility: Is marketing terminology used appropriately without 
          overwhelming technical jargon?
        - Cognitive load: Can a busy marketer quickly scan and extract key information?
        - Context relevance: Does the language connect to familiar marketing concepts?
        - Actionability: Is the tone solution-focused and implementation-ready?

        2. STRUCTURE AND FLOW:
        - Opening Impact: Does it start with business value/use case?
        - Information Hierarchy: Are headings and subheadings logically organized?
        - Navigation Efficiency: Can users quickly find specific implementation steps?
        - Visual Structure: Appropriate use of lists, callouts, and formatting?
        - Logical Progression: Does information build logically from concept to 
          implementation?

        3. COMPLETENESS AND EXAMPLES:
        - Business Impact: Clear explanation of WHY they need this feature?
        - Use Cases: WHEN to use it (specific campaign scenarios)?
        - Implementation: Can they implement WITHOUT developer support?
        - Measurement: Clear success metrics and KPIs?
        - Troubleshooting: Basic issue resolution guidance?
        - Integration: How it fits with existing workflows?

        4. STYLE GUIDELINES:
        - Customer-Obsessed Voice: Solutions-focused, empathetic language?
        - Clarity and Conciseness: Clear, concise, active voice?
        - Action-Oriented: Clear next steps and specific verbs?

        Content to analyze:
        {content[:1500]}

        Provide your analysis in this structured format:

        ## Overall Quality Score: [Excellent/Good/Needs Improvement/Poor]

        ## 1. READABILITY ASSESSMENT
        - Current Flesch Score: {flesch_score} ({self._interpret_flesch_score(flesch_score)['description']})
        - Current Gunning Fog: {gunning_fog} ({self._interpret_gunning_fog(gunning_fog)['description']})
        - Key Issues: [Bullet points]
        - Specific Recommendations: [Concrete suggestions]
        - Sample Improvements: [Before/after examples]

        ## 2. STRUCTURE & FLOW ASSESSMENT
        - Score: [Excellent/Good/Needs Improvement]
        - Structural Strengths: [What works well]
        - Gaps Identified: [What's missing or poorly organized]
        - Reorganization Suggestions: [Specific changes]

        ## 3. COMPLETENESS ASSESSMENT
        - Score: [Complete/Partially Complete/Incomplete]
        - Missing Critical Information: [List gaps]
        - Example Enhancement Opportunities: [Where to add examples]
        - Implementation Clarity: [Rate ability to execute independently]

        ## 4. STYLE GUIDELINES ASSESSMENT
        - Customer-Obsessed Tone: [Strong/Adequate/Weak] + specific feedback
        - Clarity & Conciseness: [Strong/Adequate/Weak] + specific feedback
        - Action-Oriented Language: [Strong/Adequate/Weak] + specific feedback

        ## TOP 3 PRIORITY IMPROVEMENTS
        1. [Most critical fix with specific suggestion]
        2. [Second priority with specific suggestion]
        3. [Third priority with specific suggestion]

        ## QUICK WINS
        [Easy fixes that would significantly improve quality]"""

    def _parse_assessment_scores(self, llm_response: str) -> Dict[str, Any]:
        """Parse scores from the comprehensive assessment.
        
        Args:
            llm_response: The LLM's response text
            
        Returns:
            Dict containing parsed scores
        """
        scores = {}
        try:
            # Extract overall quality score
            overall_match = re.search(r'Overall Quality Score:\s*(\w+)', llm_response)
            if overall_match:
                scores["overall_quality"] = overall_match.group(1)
            
            # Extract readability score
            readability_match = re.search(r'Score:\s*(\w+)', llm_response)
            if readability_match:
                scores["readability"] = readability_match.group(1)
            
            # Extract structure score
            structure_match = re.search(r'Score:\s*(\w+)', llm_response)
            if structure_match:
                scores["structure"] = structure_match.group(1)
            
            # Extract completeness score
            completeness_match = re.search(r'Score:\s*(\w+)', llm_response)
            if completeness_match:
                scores["completeness"] = completeness_match.group(1)
            
            # Extract style scores
            style_scores = {}
            for aspect in ["Customer-Obsessed Tone", "Clarity & Conciseness", 
                         "Action-Oriented Language"]:
                match = re.search(f'{aspect}:\s*(\w+)', llm_response)
                if match:
                    style_scores[aspect.lower().replace(" & ", "_")
                               .replace(" ", "_")] = match.group(1)
            
            scores["style"] = style_scores
            
        except Exception as e:
            print(f"Error parsing assessment scores: {e}")
            scores = {
                "overall_quality": "Needs Improvement",
                "readability": "Medium",
                "structure": "Needs Improvement",
                "completeness": "Partially Complete",
                "style": {
                    "customer_obsessed_tone": "Adequate",
                    "clarity_conciseness": "Adequate",
                    "action_oriented_language": "Adequate"
                }
            }
        
        return scores

    def _generate_verdict(self, scores: Dict[str, Any]) -> str:
        """Generate quick verdict based on assessment scores.
        
        Args:
            scores: Dictionary containing assessment scores
            
        Returns:
            str: Verdict message
        """
        overall = scores.get("overall_quality", "Needs Improvement")
        
        if overall == "Excellent":
            return "Excellent documentation quality"
        elif overall == "Good":
            return "Good documentation with minor improvements needed"
        elif overall == "Needs Improvement":
            return "Documentation needs significant improvement"
        else:
            return "Poor documentation quality, major revision required"

    def _quick_clean(self, content: str) -> str:
        """Fast content cleaning for readability analysis.
        
        Args:
            content: The content to clean
            
        Returns:
            str: Cleaned content
        """
        # Remove console logs and browser messages
        content = re.sub(r'\[.*?\]\s*".*?"', '', content)
        content = re.sub(r'Waiting for content to load\.\.\.', '', content)
        content = re.sub(r'Trying selector:.*', '', content)
        content = re.sub(r'Found content using selector:.*', '', content)
        content = re.sub(r'Successfully extracted content.*', '', content)
        
        # Remove common markup quickly
        content = re.sub(r'<[^>]+>', '', content)  # HTML tags
        content = re.sub(r'#{1,6}\s*', '', content)  # Headers
        content = re.sub(r'[\*_]{1,2}([^\*_]+)[\*_]{1,2}', r'\1', content)  # Bold/italic
        content = re.sub(r'```[^`]*```', '', content)  # Code blocks
        content = re.sub(r'`[^`]+`', '', content)  # Inline code
        content = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', content)  # Links
        
        # Clean bullets but keep structure
        content = re.sub(r'^[\s]*[-*+•]\s+', '', content, flags=re.MULTILINE)
        content = re.sub(r'^[\s]*\d+\.\s+', '', content, flags=re.MULTILINE)
        
        # Remove multiple newlines and spaces
        content = re.sub(r'\n\s*\n', '\n', content)
        content = re.sub(r'\s+', ' ', content)
        
        # Remove any remaining console-like messages
        content = re.sub(r'\[.*?\]', '', content)
        content = re.sub(r'\(.*?\)', '', content)
        
        # Remove duplicate sentences
        sentences = content.split('.')
        unique_sentences = []
        seen = set()
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and sentence not in seen:
                seen.add(sentence)
                unique_sentences.append(sentence)
        
        # Final whitespace normalization
        content = '. '.join(unique_sentences)
        content = ' '.join(content.split())
        
        return content.strip()

    def analyze_structure(self, content: str) -> Dict[str, Any]:
        """Analyze document structure using LLM assessment.
        
        Args:
            content: The content to analyze
            
        Returns:
            Dict containing structure analysis results
        """
        try:
            # Analyze paragraphs
            paragraphs = [p for p in content.split('\n\n') if p.strip()]
            long_paragraphs = [
                p for p in paragraphs 
                if len(p.split()) > self.MAX_PARAGRAPH_LENGTH
            ]
            
            # Analyze lists
            bullet_count = len(re.findall(r'^[\s]*[-*+•]\s+', content, re.MULTILINE))
            numbered_count = len(re.findall(r'^[\s]*\d+\.\s+', content, re.MULTILINE))
            
            prompt = f"""Analyze the structure and organization of this documentation. Consider:
            1. Logical flow and progression of information
            2. Use of headings and sections
            3. Information hierarchy
            4. Navigation ease
            5. Specific examples of where structure could be improved
            
            Content:
            {content[:2000]}
            """
            
            llm_assessment = self.call_llm(prompt)
            
            # Parse structure scores from LLM response
            structure_scores = self._parse_structure_scores(llm_assessment)
            
            return {
                "llm_assessment": llm_assessment,
                "structure_scores": structure_scores,
                "long_paragraphs": len(long_paragraphs),
                "list_analysis": {
                    "bullets": bullet_count,
                    "numbered": numbered_count,
                    "total_lists": bullet_count + numbered_count
                }
            }
        except Exception as e:
            print(f"Error in structure analysis: {e}")
            return {
                "llm_assessment": f"Error: {str(e)}",
                "structure_scores": {},
                "long_paragraphs": 0,
                "list_analysis": {
                    "bullets": 0,
                    "numbered": 0,
                    "total_lists": 0
                }
            }

    def _parse_structure_scores(self, llm_response: str) -> Dict[str, float]:
        """Parse structure scores from LLM response.
        
        Args:
            llm_response: The LLM's response text
            
        Returns:
            Dict containing parsed structure scores
        """
        scores = {}
        try:
            # Extract scores with regex
            patterns = {
                "logical_flow": r'LOGICAL FLOW:\s*(\d+(?:\.\d+)?)',
                "headings": r'HEADINGS:\s*(\d+(?:\.\d+)?)',
                "hierarchy": r'HIERARCHY:\s*(\d+(?:\.\d+)?)',
                "navigation": r'NAVIGATION:\s*(\d+(?:\.\d+)?)',
                "overall": r'OVERALL:\s*(\d+(?:\.\d+)?)'
            }
            
            for key, pattern in patterns.items():
                match = re.search(pattern, llm_response, re.IGNORECASE)
                if match:
                    scores[key] = float(match.group(1))
            
            # Calculate overall if missing
            if "overall" not in scores and len(scores) >= 3:
                scores["overall"] = round(sum(scores.values()) / len(scores), 1)
                
        except Exception:
            # Fallback scores if parsing fails
            scores = {
                "logical_flow": 5,
                "headings": 5,
                "hierarchy": 5,
                "navigation": 5,
                "overall": 5
            }
        
        return scores

    def analyze_completeness(self, content: str) -> Dict[str, Any]:
        try:
            # Analyze examples
            example_keywords = ["for example", "e.g.", "such as", "like", "instance"]
            examples = []
            for keyword in example_keywords:
                if keyword in content.lower():
                    start = content.lower().find(keyword)
                    example = content[start:start+200]  # Get context around example
                    examples.append({
                        "context": example,
                        "keyword": keyword,
                        "suggestion": "Consider expanding this example with more details."
                    })
            
            prompt = f"""Assess the completeness of this documentation. Consider:
            1. Coverage of necessary information
            2. Use of examples and use cases
            3. Missing or unclear information
            4. Specific suggestions for additional examples
            5. Concrete examples of what information is missing
            
            Content:
            {content[:2000]}
            """
            
            llm_assessment = self.call_llm(prompt)
            return {
                "examples": examples,
                "llm_assessment": llm_assessment
            }
        except Exception as e:
            print(f"Error in completeness analysis: {e}")
            return {
                "examples": [],
                "llm_assessment": f"Error: {str(e)}"
            }

    def analyze_style_guidelines(self, content: str) -> Dict[str, Any]:
        """Analyze content against Microsoft Style Guide principles using LLM assessment.
        
        Args:
            content: The content to analyze
            
        Returns:
            Dict containing style analysis results
        """
        try:
            prompt = f"""Analyze this documentation against Microsoft Style Guide principles. Consider:

1. VOICE AND TONE (1-10):
   - Customer-focused language
   - Conversational tone
   - Use of contractions
   - Professional but friendly style

2. CLARITY AND CONCISENESS (1-10):
   - Clear and direct language
   - Sentence length and complexity
   - Word choice and jargon
   - Active vs passive voice

3. ACTION-ORIENTED LANGUAGE (1-10):
   - Use of strong action verbs
   - Clear instructions
   - Step-by-step guidance
   - Task-oriented language

4. MICROSOFT-SPECIFIC STYLE (1-10):
   - Capitalization rules
   - Punctuation usage
   - List formatting
   - Heading style

Content:
{content[:2000]}

Respond in this format:
VOICE AND TONE: [score]/10 - [analysis]
CLARITY: [score]/10 - [analysis]
ACTION-ORIENTED: [score]/10 - [analysis]
MICROSOFT STYLE: [score]/10 - [analysis]
OVERALL: [average score]/10 - [summary]

Also list any specific style issues found in this format:
ISSUES:
- [Category]: [Description] (Priority: High/Medium/Low)"""

            llm_assessment = self.call_llm(prompt)
            
            # Parse style scores
            style_scores = self._parse_style_scores(llm_assessment)
            
            # Parse style issues
            issues = self._parse_style_issues(llm_assessment)
            
            return {
                "llm_assessment": llm_assessment,
                "style_scores": style_scores,
                "total_issues": len(issues),
                "high_priority_issues": len([i for i in issues if i["priority"] == "High"]),
                "medium_priority_issues": len([i for i in issues if i["priority"] == "Medium"]),
                "low_priority_issues": len([i for i in issues if i["priority"] == "Low"]),
                "issues_by_category": self._categorize_issues(issues),
                "detailed_issues": issues,
                "summary": self._generate_style_summary(style_scores)
            }
            
        except Exception as e:
            print(f"Error in style analysis: {e}")
            return {
                "error": f"Analysis failed: {str(e)}",
                "llm_assessment": f"Error: {str(e)}",
                "style_scores": {},
                "total_issues": 0,
                "high_priority_issues": 0,
                "medium_priority_issues": 0,
                "low_priority_issues": 0,
                "issues_by_category": {},
                "detailed_issues": [],
                "summary": "Error occurred during analysis"
            }

    def _parse_style_issues(self, llm_response: str) -> List[Dict[str, str]]:
        """Parse style issues from LLM response.
        
        Args:
            llm_response: The LLM's response text
            
        Returns:
            List of dictionaries containing parsed issues
        """
        issues = []
        try:
            # Extract issues section
            issues_section = re.search(r'ISSUES:(.*?)(?=\n\n|$)', 
                                     llm_response, re.DOTALL)
            if issues_section:
                # Parse each issue
                for line in issues_section.group(1).split('\n'):
                    if line.strip().startswith('-'):
                        # Extract category, description, and priority
                        match = re.match(r'- \[?([^\]]+)\]?: (.*?) \(Priority: (High|Medium|Low)\)', 
                                       line.strip('- '))
                        if match:
                            issues.append({
                                "category": match.group(1).strip(),
                                "description": match.group(2).strip(),
                                "priority": match.group(3)
                            })
        except Exception as e:
            print(f"Error parsing style issues: {e}")
        
        return issues

    def _categorize_issues(self, issues: List[Dict[str, str]]) -> Dict[str, int]:
        """Categorize issues by their category.
        
        Args:
            issues: List of issue dictionaries
            
        Returns:
            Dictionary with category counts
        """
        categories = {}
        for issue in issues:
            category = issue["category"]
            categories[category] = categories.get(category, 0) + 1
        return categories

    def _parse_style_scores(self, llm_response: str) -> Dict[str, float]:
        """Parse style scores from LLM response.
        
        Args:
            llm_response: The LLM's response text
            
        Returns:
            Dict containing parsed style scores
        """
        scores = {}
        try:
            # Extract scores with regex
            patterns = {
                "voice_tone": r'VOICE AND TONE:\s*(\d+(?:\.\d+)?)',
                "clarity": r'CLARITY:\s*(\d+(?:\.\d+)?)',
                "action_oriented": r'ACTION-ORIENTED:\s*(\d+(?:\.\d+)?)',
                "microsoft_style": r'MICROSOFT STYLE:\s*(\d+(?:\.\d+)?)',
                "overall": r'OVERALL:\s*(\d+(?:\.\d+)?)'
            }
            
            for key, pattern in patterns.items():
                match = re.search(pattern, llm_response, re.IGNORECASE)
                if match:
                    scores[key] = float(match.group(1))
            
            # Calculate overall if missing
            if "overall" not in scores and len(scores) >= 3:
                scores["overall"] = round(sum(scores.values()) / len(scores), 1)
                
        except Exception as e:
            print(f"Error parsing style scores: {e}")
            # Fallback scores if parsing fails
            scores = {
                "voice_tone": 5,
                "clarity": 5,
                "action_oriented": 5,
                "microsoft_style": 5,
                "overall": 5
            }
        
        return scores

    def _generate_style_summary(self, scores: Dict[str, float]) -> str:
        """Generate a summary of the style analysis.
        
        Args:
            scores: Dictionary containing style scores
            
        Returns:
            str: Summary of style analysis
        """
        if not scores:
            return "Unable to generate style summary."
            
        overall = scores.get("overall", 0)
        if overall >= 8:
            return "Excellent adherence to Microsoft Style Guide principles."
        elif overall >= 6:
            return "Good adherence to style guidelines with some areas for improvement."
        elif overall >= 4:
            return "Moderate adherence to style guidelines, needs significant improvement."
        else:
            return "Poor adherence to style guidelines, requires major revision."

    def _compile_report(self, url: str, readability: Dict, structure: Dict, 
                       completeness: Dict, style: Dict) -> Dict[str, Any]:
        """Compile the final report from all analysis components."""
        try:
            # Calculate scores with error handling
            readability_score = self._calculate_score(readability.get("llm_assessment", ""))
            structure_score = self._calculate_score(structure.get("llm_assessment", ""))
            completeness_score = self._calculate_score(completeness.get("llm_assessment", ""))
            style_score = self._calculate_score(style.get("summary", ""))
            
            # Extract suggestions with error handling
            readability_suggestions = self._extract_suggestions(readability.get("llm_assessment", ""))
            structure_suggestions = self._extract_suggestions(structure.get("llm_assessment", ""))
            completeness_suggestions = self._extract_suggestions(completeness.get("llm_assessment", ""))
            style_suggestions = self._extract_suggestions(style.get("summary", ""))
            
            # Calculate overall score
            overall_score = (readability_score + structure_score + 
                           completeness_score + style_score) / 4
            
            # Generate summary
            summary = {
                "overall_score": round(overall_score, 1),
                "aspect_scores": {
                    "readability": readability_score,
                    "structure": structure_score,
                    "completeness": completeness_score,
                    "style": style_score
                },
                "critical_issues": [
                    *readability_suggestions.get("critical", []),
                    *structure_suggestions.get("critical", []),
                    *completeness_suggestions.get("critical", []),
                    *style_suggestions.get("critical", [])
                ],
                "quick_wins": [
                    *readability_suggestions.get("quick_wins", []),
                    *structure_suggestions.get("quick_wins", []),
                    *completeness_suggestions.get("quick_wins", []),
                    *style_suggestions.get("quick_wins", [])
                ]
            }
            
            # Compile report with error handling
            report = {
                "url": url,
                "summary": summary,
                "readability": {
                    "score": readability_score,
                    "flesch_score": readability.get("flesch_score", 0),
                    "gunning_fog": readability.get("gunning_fog", 0),
                    "complex_sentences": readability.get("complex_sentences", 0),
                    "llm_assessment": readability.get("llm_assessment", "No assessment available"),
                    "suggestions": readability_suggestions
                },
                "structure": {
                    "score": structure_score,
                    "long_paragraphs": structure.get("long_paragraphs", 0),
                    "list_analysis": structure.get("list_analysis", {
                        "bullets": 0,
                        "numbered": 0,
                        "total_lists": 0
                    }),
                    "llm_assessment": structure.get("llm_assessment", "No assessment available"),
                    "suggestions": structure_suggestions
                },
                "completeness": {
                    "score": completeness_score,
                    "examples": completeness.get("examples", []),
                    "llm_assessment": completeness.get("llm_assessment", "No assessment available"),
                    "suggestions": completeness_suggestions
                },
                "style_guidelines": {
                    "score": style_score,
                    "total_issues": style.get("total_issues", 0),
                    "high_priority_issues": style.get("high_priority_issues", 0),
                    "medium_priority_issues": style.get("medium_priority_issues", 0),
                    "low_priority_issues": style.get("low_priority_issues", 0),
                    "issues_by_category": style.get("issues_by_category", {}),
                    "detailed_issues": style.get("detailed_issues", []),
                    "summary": style.get("summary", "No summary available"),
                    "llm_assessment": style.get("llm_assessment", "No assessment available")
                }
            }
            
            return report
            
        except Exception as e:
            print(f"Error compiling report: {e}")
            return {
                "url": url,
                "error": f"Error compiling report: {str(e)}",
                "status": "error"
            }

    def generate_report(self, url: str, content: str) -> Dict[str, Any]:
        """Generate a comprehensive analysis report for the documentation."""
        if not self.validate_url(url):
            return {
                "error": f"Invalid URL format: {url}",
                "status": "error",
                "url": url
            }

        if content.startswith("Error:"):
            return {
                "error": content,
                "status": "error",
                "url": url
            }

        try:
            # Perform analysis with error handling
            readability = self.analyze_readability(content)
            if "error" in readability:
                print(f"Readability analysis error: {readability['error']}")
                readability = {"llm_assessment": "Error in readability analysis"}

            structure = self.analyze_structure(content)
            if "error" in structure:
                print(f"Structure analysis error: {structure['error']}")
                structure = {"llm_assessment": "Error in structure analysis"}

            completeness = self.analyze_completeness(content)
            if "error" in completeness:
                print(f"Completeness analysis error: {completeness['error']}")
                completeness = {"llm_assessment": "Error in completeness analysis"}

            style = self.analyze_style_guidelines(content)
            if "error" in style:
                print(f"Style analysis error: {style['error']}")
                style = {"llm_assessment": "Error in style analysis"}
            
            # Compile report
            report = self._compile_report(url, readability, structure, completeness, style)
            
            # Save the report
            self.save_analysis(url, report)
            
            report["status"] = "success"
            return report
            
        except Exception as e:
            error_msg = f"Error generating report: {str(e)}"
            print(error_msg)
            return {
                "error": error_msg,
                "status": "error",
                "url": url
            }

    def _calculate_score(self, assessment: str) -> int:
        """Calculate a score (1-10) based on the assessment text."""
        if "Error" in assessment:
            return 0
            
        positive_indicators = ["good", "excellent", "clear", "well", 
                             "comprehensive", "detailed"]
        negative_indicators = ["poor", "unclear", "missing", "incomplete", 
                             "lacks", "needs improvement"]
        
        score = 5  # Start with neutral score
        for indicator in positive_indicators:
            if indicator in assessment.lower():
                score += 1
        for indicator in negative_indicators:
            if indicator in assessment.lower():
                score -= 1
        
        return max(1, min(10, score))

    def _extract_suggestions(self, assessment: str) -> Dict[str, list]:
        """Extract and categorize suggestions from the assessment."""
        if "Error" in assessment:
            return {"critical": [], "quick_wins": [], "other": []}
            
        suggestions = []
        current_section = ""
        
        for line in assessment.split('\n'):
            if "**" in line:
                current_section = line.strip('*').strip()
            elif line.strip().startswith(('*', '-')):
                suggestion = line.strip('*').strip('-').strip()
                if suggestion:
                    suggestions.append({
                        "text": suggestion,
                        "section": current_section,
                        "priority": "other"
                    })
        
        return self._categorize_suggestions(suggestions)

    def _categorize_suggestions(self, suggestions: List[Dict]) -> Dict[str, list]:
        """Categorize suggestions into critical, quick wins, and other."""
        critical_keywords = ["missing", "error", "must", "required", 
                           "critical", "essential"]
        quick_win_keywords = ["simple", "easy", "quick", "minor", 
                            "small", "basic"]
        
        categorized = {
            "critical": [],
            "quick_wins": [],
            "other": []
        }
        
        for suggestion in suggestions:
            text = suggestion["text"].lower()
            if any(keyword in text for keyword in critical_keywords):
                suggestion["priority"] = "critical"
                categorized["critical"].append(suggestion)
            elif any(keyword in text for keyword in quick_win_keywords):
                suggestion["priority"] = "quick_win"
                categorized["quick_wins"].append(suggestion)
            else:
                categorized["other"].append(suggestion)
        
        return categorized

    def save_analysis(self, url: str, report: Dict[str, Any]) -> None:
        """Save the analysis report to a text file.
        
        Args:
            url: The URL that was analyzed
            report: The analysis report to save
        """
        try:
            # Create outputs directory if it doesn't exist
            os.makedirs("outputs", exist_ok=True)
            
            # Generate filename from URL
            filename = url.split("/")[-1].replace("#", "_")
            filepath = f"outputs/{filename}_analysis.txt"
            
            # Format the report as a readable text file
            with open(filepath, "w", encoding="utf-8") as f:
                # Write header
                f.write(f"Documentation Analysis Report\n")
                f.write(f"URL: {url}\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 80 + "\n\n")
                
                # Write summary
                f.write("SUMMARY\n")
                f.write("-" * 40 + "\n")
                f.write(f"Overall Score: {report['summary']['overall_score']}/10\n\n")
                
                f.write("Aspect Scores:\n")
                for aspect, score in report['summary']['aspect_scores'].items():
                    f.write(f"- {aspect.title()}: {score}/10\n")
                f.write("\n")
                
                if report['summary']['critical_issues']:
                    f.write("Critical Issues:\n")
                    for issue in report['summary']['critical_issues']:
                        f.write(f"- {issue}\n")
                    f.write("\n")
                
                if report['summary']['quick_wins']:
                    f.write("Quick Wins:\n")
                    for win in report['summary']['quick_wins']:
                        f.write(f"- {win}\n")
                    f.write("\n")
                
                # Write readability analysis
                f.write("READABILITY ANALYSIS\n")
                f.write("-" * 40 + "\n")
                f.write(f"Word Count: {report['readability'].get('word_count', 'N/A')}\n")
                
                flesch = report['readability'].get('flesch_score', {})
                if flesch:
                    f.write(f"\nFlesch Reading Ease Score: {flesch.get('score', 'N/A')}\n")
                    f.write(f"Grade Level: {flesch.get('grade_level', 'N/A')}\n")
                    f.write(f"Description: {flesch.get('description', 'N/A')}\n")
                    f.write(f"Writing Style: {flesch.get('writing_style', 'N/A')}\n")
                    f.write(f"Target Audience: {flesch.get('target_audience', 'N/A')}\n")
                
                fog = report['readability'].get('gunning_fog', {})
                if fog:
                    f.write(f"\nGunning Fog Index: {fog.get('score', 'N/A')}\n")
                    f.write(f"Grade Level: {fog.get('grade_level', 'N/A')}\n")
                    f.write(f"Description: {fog.get('description', 'N/A')}\n")
                    f.write(f"Writing Style: {fog.get('writing_style', 'N/A')}\n")
                    f.write(f"Target Audience: {fog.get('target_audience', 'N/A')}\n")
                
                f.write(f"\nComplex Sentences: {report['readability'].get('complex_sentences', 'N/A')}\n")
                
                if report['readability'].get('llm_assessment'):
                    f.write("\nDetailed Assessment:\n")
                    f.write(report['readability']['llm_assessment'] + "\n")
                
                # Write structure analysis
                f.write("\nSTRUCTURE ANALYSIS\n")
                f.write("-" * 40 + "\n")
                f.write(f"Score: {report['structure'].get('score', 'N/A')}/10\n")
                f.write(f"Long Paragraphs: {report['structure'].get('long_paragraphs', 'N/A')}\n")
                
                list_analysis = report['structure'].get('list_analysis', {})
                if list_analysis:
                    f.write("\nList Analysis:\n")
                    f.write(f"- Bullet Points: {list_analysis.get('bullets', 'N/A')}\n")
                    f.write(f"- Numbered Lists: {list_analysis.get('numbered', 'N/A')}\n")
                    f.write(f"- Total Lists: {list_analysis.get('total_lists', 'N/A')}\n")
                
                if report['structure'].get('llm_assessment'):
                    f.write("\nDetailed Assessment:\n")
                    f.write(report['structure']['llm_assessment'] + "\n")
                
                # Write completeness analysis
                f.write("\nCOMPLETENESS ANALYSIS\n")
                f.write("-" * 40 + "\n")
                f.write(f"Score: {report['completeness'].get('score', 'N/A')}/10\n")
                
                examples = report['completeness'].get('examples', [])
                if examples:
                    f.write("\nExample Enhancements:\n")
                    for example in examples:
                        f.write(f"- Context: {example.get('context', 'N/A')}\n")
                        f.write(f"  Keyword: {example.get('keyword', 'N/A')}\n")
                        f.write(f"  Suggestion: {example.get('suggestion', 'N/A')}\n")
                
                if report['completeness'].get('llm_assessment'):
                    f.write("\nDetailed Assessment:\n")
                    f.write(report['completeness']['llm_assessment'] + "\n")
                
                # Write style guidelines analysis
                f.write("\nSTYLE GUIDELINES ANALYSIS\n")
                f.write("-" * 40 + "\n")
                f.write(f"Score: {report['style_guidelines'].get('score', 'N/A')}/10\n")
                
                f.write("\nIssues by Priority:\n")
                f.write(f"- High Priority: {report['style_guidelines'].get('high_priority_issues', 'N/A')}\n")
                f.write(f"- Medium Priority: {report['style_guidelines'].get('medium_priority_issues', 'N/A')}\n")
                f.write(f"- Low Priority: {report['style_guidelines'].get('low_priority_issues', 'N/A')}\n")
                f.write(f"- Total Issues: {report['style_guidelines'].get('total_issues', 'N/A')}\n")
                
                detailed_issues = report['style_guidelines'].get('detailed_issues', [])
                if detailed_issues:
                    f.write("\nDetailed Issues:\n")
                    for issue in detailed_issues:
                        f.write(f"- [{issue.get('priority', 'N/A')}] {issue.get('category', 'N/A')}: {issue.get('description', 'N/A')}\n")
                
                if report['style_guidelines'].get('summary'):
                    f.write("\nSummary:\n")
                    f.write(report['style_guidelines']['summary'] + "\n")
                
                if report['style_guidelines'].get('llm_assessment'):
                    f.write("\nDetailed Assessment:\n")
                    f.write(report['style_guidelines']['llm_assessment'] + "\n")
            
            print(f"\nAnalysis saved to: {filepath}")
            
        except Exception as e:
            print(f"Error saving analysis: {e}") 