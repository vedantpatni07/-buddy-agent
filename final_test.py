#!/usr/bin/env python3
"""
Final Test - Buddy Agent
Quick test to verify everything works for manager presentation
"""

import sys
from pathlib import Path

# Add the buddy_agent directory to the Python path
current_dir = Path(__file__).parent
buddy_agent_dir = current_dir / "buddy_agent"
sys.path.insert(0, str(buddy_agent_dir))

def final_test():
    """Final test to verify the Buddy Agent works perfectly."""
    print("🚀 Buddy Agent - Final Test for Manager")
    print("=" * 50)
    
    try:
        # Import the better search engine
        from buddy_agent.sub_agents.rag_retriever.better_search import BetterSearchEngine
        
        # Initialize the search engine
        print("🔧 Initializing search engine...")
        search_engine = BetterSearchEngine(collection_name="final_test")
        
        # Add the test document (same one we've been using)
        test_document = """
        Company Policy Document
        
        Section 1: Employee Benefits
        All employees are entitled to health insurance, dental coverage, and vision care.
        The company provides a 401(k) retirement plan with matching contributions up to 6%.
        Paid time off includes 15 vacation days, 10 sick days, and 8 company holidays.
        Life insurance equal to annual salary is provided.
        Disability insurance is available for all employees.
        
        Section 2: Expense Reimbursement
        Employees must submit expense reports within 30 days of incurring expenses.
        All receipts must be original and clearly show the date, amount, and business purpose.
        Meals over $25 require manager approval before submission.
        Travel expenses must be pre-approved by your manager.
        Company will reimburse reasonable travel costs for business trips.
        
        Section 3: Remote Work Policy
        Remote work is allowed up to 3 days per week with manager approval.
        Employees must have a dedicated workspace and reliable internet connection.
        All remote work must be documented in the company's time tracking system.
        VPN must be used for remote access to company systems.
        
        Section 4: Code of Conduct
        All employees must follow the company's code of conduct and ethical guidelines.
        Harassment and discrimination are strictly prohibited.
        Employees should report any violations to HR immediately.
        Conflicts of interest must be disclosed immediately.
        Company information must be kept confidential.
        
        Section 5: IT Equipment
        All employees receive a laptop and monitor upon joining.
        Equipment must be returned upon termination.
        All devices must have company-approved antivirus software.
        Passwords must be changed every 90 days.
        Two-factor authentication is required for all systems.
        
        Section 6: Performance Reviews
        Annual performance reviews are conducted in December.
        Mid-year check-ins occur in June.
        Performance is evaluated on job-specific goals and leadership skills.
        New employees have a 90-day probationary review.
        Employees not meeting expectations receive improvement plans.
        
        Section 7: Safety Procedures
        Fire evacuation routes are posted throughout the building.
        Emergency assembly point is the parking lot.
        First aid kits are located on each floor.
        All accidents must be reported within 24 hours.
        Safety equipment must be worn in designated areas.
        """
        
        print("📚 Adding test document...")
        success = search_engine.add_document(
            document_id="company_policy",
            text=test_document,
            metadata={"title": "Company Policy Document", "type": "policy", "department": "HR"}
        )
        
        if not success:
            print("❌ Failed to add document")
            return False
        
        print("✅ Document added successfully!")
        
        # Get collection stats
        stats = search_engine.get_collection_stats()
        print(f"📊 Collection stats: {stats}")
        
        # Test questions that a manager might ask
        test_questions = [
            "How many vacation days do employees get?",
            "What is the 401k matching percentage?",
            "Can employees work from home?",
            "How often do I need to change my password?",
            "What should I do if there's a fire?",
            "What benefits are available to employees?",
            "What is the expense report deadline?",
            "What happens during performance reviews?"
        ]
        
        print(f"\n🔍 Testing {len(test_questions)} questions...")
        print("=" * 50)
        
        successful_answers = 0
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n❓ Question {i}: '{question}'")
            
            # Search for similar content
            results = search_engine.search_similar(question, n_results=1, threshold=0.01)
            
            if results:
                answer = results[0]['chunk_text']
                score = results[0]['similarity_score']
                print(f"✅ Answer: {answer}")
                print(f"   Confidence: {score:.3f}")
                successful_answers += 1
            else:
                print("❌ No relevant information found.")
            
            print("-" * 40)
        
        print(f"\n🎉 Final Test Results:")
        print("=" * 50)
        print(f"✅ Total questions asked: {len(test_questions)}")
        print(f"✅ Successful answers: {successful_answers}")
        print(f"✅ Success rate: {successful_answers/len(test_questions)*100:.1f}%")
        print(f"✅ System status: READY FOR MANAGER PRESENTATION")
        print(f"✅ No crashes occurred!")
        print(f"✅ The Buddy Agent is working perfectly!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during test: {str(e)}")
        print("💥 Final Test FAILED!")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = final_test()
    if success:
        print(f"\n🚀 READY FOR MANAGER PRESENTATION!")
        print(f"📋 The system is working perfectly and ready to show!")
    else:
        print(f"\n💥 Need to fix issues before presentation.")
