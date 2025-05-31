# Documentation Analyzer

A comprehensive tool for analyzing and improving technical documentation quality, with a focus on readability, structure, completeness, and style guidelines.

## Features

- **Readability Analysis**
  - Flesch Reading Ease Score
  - Gunning Fog Index
  - Complex sentence detection
  - Grade level assessment
  - Target audience analysis

- **Structure Analysis**
  - Logical flow assessment
  - Heading hierarchy
  - List analysis (bullets and numbered)
  - Paragraph length analysis
  - Navigation ease

- **Completeness Analysis**
  - Example coverage
  - Missing information detection
  - Implementation clarity
  - Use case coverage

- **Style Guidelines Analysis**
  - Microsoft Style Guide compliance
  - Voice and tone assessment
  - Clarity and conciseness
  - Action-oriented language
  - Technical writing best practices

## Setup Instructions

1. **Prerequisites**
   ```bash
   Python 3.8+
   Chrome Browser (for Selenium)
   Gemini API
   ```

2. **Installation**
   ```bash
   # Clone the repository
   git clone [repository-url]
   cd documentation-analyzer

   # Install dependencies
   pip install -r requirements.txt
   ```

3. **Environment Setup**
   - Create a `.env` file in the project root
   - Add your Gemini API key:
     ```
     GEMINI_API_KEY=your_api_key_here
     ```

4. **Running the Analyzer**
   ```bash
   python src/run_analyzer.py
   ```

## Design Choices and Approach


### 1. Style Guidelines Implementation Approach

For the style guidelines, my approach was to understand the main principles of technical writing. I studied the Microsoft Style Guide, specifically focusing on the top 10 tips for mastering Microsoft style and voice. Based on this research, I implemented a comprehensive style analysis function that incorporates these key Microsoft Style Guide principles:

a. **Voice and Tone (Customer-focused, clear, conversational)**
   - Detects overly formal language and suggests conversational alternatives
   - Identifies missing contractions (Microsoft recommends "it's, you'll, you're, we're, let's")
   - Promotes friendly, approachable tone

b. **Clarity and Conciseness ("Use bigger ideas, fewer words")**
   - Flags overly long sentences (Microsoft prefers shorter, scannable content)
   - Identifies wordy phrases that can be simplified
   - Detects passive voice and suggests active voice alternatives
   - Flags unexplained technical jargon

c. **Action-Oriented Language ("Most of the time, start each statement with a verb")**
   - Identifies weak instruction language like "you can" and suggests direct action verbs
   - Checks if instructions start with strong action verbs
   - Promotes clear, actionable guidance

d. **Microsoft-Specific Style Rules**
   - Enforces sentence-style capitalization (avoiding Title Case)
   - Checks for unnecessary periods in headings
   - Ensures Oxford comma usage in lists

### 2. Prompt Engineering Approach

My second approach focused on improving prompts by defining a clear MoEngage Marketer Persona. I started by understanding our primary audience:

**Primary Audience Definition:**
- Role: Growth marketers, product marketers, campaign managers at consumer brands
- Technical Level: Semi-technical (understands marketing technology but isn't a developer)
- Goals: Drive engagement, retention, conversions through omnichannel campaigns
- Pain Points: Needs clear, actionable guidance to implement features quickly
- Context: Works with customer engagement platforms, personalization, segmentation

Based on this understanding, I developed a comprehensive prompt engineering framework:

a. **Persona Definition in Prompts:**
   - Analyzes content for growth marketers at consumer brands
   - Considers basic technical knowledge without developer expertise
   - Focuses on quick implementation and business results
   - Emphasizes clear examples and step-by-step guidance
   - Accounts for tight deadlines and multiple campaigns

b. **Style Guide Principles:**
   - Customer-obsessed tone: Direct, helpful, solution-focused
   - Action-oriented: Each section leads to clear next steps
   - Business-outcome focused: Connects features to marketing goals
   - Concise but complete: Comprehensive without overwhelming

c. **Structure Expectations:**
   - Clear introduction explaining business value
   - Prerequisites/setup requirements upfront
   - Step-by-step implementation with screenshots
   - Real-world examples with actual use cases
   - Troubleshooting common issues
   - Next steps or related features

d. **Completeness Criteria:**
   - Understanding WHY they need the feature (business impact)
   - Knowing WHEN to use it (specific scenarios)
   - Implementation WITHOUT additional help
   - Success measurement (metrics/KPIs)
   - Basic troubleshooting independence

I implemented this through strategic prompt designs:

a. **Persona-Driven Assessment:**
   - Every criterion filtered through the MoEngage marketer persona
   - Highly relevant evaluations
   - Marketing-focused analysis

b. **Business-Context Integration:**
   - Connects technical documentation to marketing success
   - Aligns with MoEngage's customer-obsessed approach
   - Focuses on campaign outcomes

c. **Actionable Output Structure:**
   - Specific, implementable recommendations
   - Beyond just scoring
   - Clear improvement paths

d. **MoEngage-Specific Elements:**
   - Campaign workflow integration
   - Feature interconnections
   - Marketing KPI focus
   - Customer engagement terminology

e. **Scalable Framework:**
   - Master template for different documentation types
   - Adjustable context
   - Consistent evaluation approach

### 3. Readability Scoring Methodology
- **Flesch-Kincaid Analysis**:
  - Sentence length weighting
  - Word length consideration
  - Technical term handling
  - Context-aware scoring

- **Gunning Fog Index**:
  - Complex word identification
  - Sentence structure analysis
  - Technical content adjustment
  - Readability level mapping

- **Custom Scoring Factors**:
  - Technical term density
  - Code example impact
  - Domain-specific terminology
  - Audience level adjustment

- **Score Interpretation**:
  - Industry-specific benchmarks
  - Technical content allowance
  - Target audience alignment
  - Improvement suggestions

## Assumptions

1. **Content Structure**
   - Documentation follows standard web format
   - Content is primarily text-based
   - Technical terms are necessary
   - Target audience is technical but not expert

2. **Style Guidelines**
   - Microsoft Style Guide is applicable
   - Technical accuracy is paramount
   - Clarity is more important than brevity
   - Examples enhance understanding

3. **Analysis Scope**
   - Content is in English
   - Technical terms are domain-specific
   - Documentation is for end-users
   - Implementation details are important

## Challenges and Solutions

### 1. Content Extraction
**Challenge**: Extracting clean content from various documentation formats and handling website protection mechanisms
**Solution**: 
- Implemented Selenium with Chrome WebDriver for reliable content extraction
- Added multiple CSS selectors for different documentation formats
- Implemented content validation and cleaning pipeline
- Added fallback mechanisms for different page structures
- Handled dynamic content loading and JavaScript-rendered pages

### 2. Readability Metrics
**Challenge**: Balancing technical accuracy with readability while handling domain-specific terminology
**Solution**:
- Custom interpretation of Flesch and Gunning Fog scores
- Context-aware assessment for technical content
- Special handling for code examples and technical terms
- Industry-specific readability targets
- Detailed explanations for score adjustments

### 3. Style Analysis
**Challenge**: Applying Microsoft Style Guide consistently while maintaining technical accuracy
**Solution**:
- Structured assessment framework with priority levels
- Categorized issues by severity (High, Medium, Low)
- Specific examples and suggestions for each issue
- Clear improvement paths with actionable steps
- Balance between style guidelines and technical requirements

### 4. Performance
**Challenge**: Processing large documentation sets efficiently
**Solution**:
- Implemented content caching
- Optimized content cleaning pipeline
- Added batch processing capability
- Used efficient data structures
- Implemented parallel processing where possible

### 5. Output Format
**Challenge**: Creating consistent, readable output format
**Solution**:
- Designed structured text file format
- Implemented clear section separation
- Added detailed metrics and scores
- Included actionable suggestions
- Created priority-based improvement lists


## Example Outputs

The analyzer generates detailed text files in the `outputs` directory for each analyzed URL. Each output file contains a comprehensive analysis report.

### Output Files
- Located in the `outputs` directory
- Named based on the analyzed URL (e.g., `All-segments-analysis.txt`)
- Contains complete analysis in a single, well-formatted text file
- Includes all metrics, scores, and detailed assessments

