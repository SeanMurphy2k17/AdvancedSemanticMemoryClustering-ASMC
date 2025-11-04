#!/usr/bin/env python3
"""Fix database path references"""

with open('dual_database_manager.py', 'r') as f:
    content = f.read()

content = content.replace('self.knowledge_db.db_path', 'self.knowledge_db.db_manager.db_path')
content = content.replace('self.experience_db.db_path', 'self.experience_db.db_manager.db_path')

with open('dual_database_manager.py', 'w') as f:
    f.write(content)

print('✅ Fixed database path references') 