"""Metric quantities explicitly associated with altitude or distance travelled, in metres."""
import re
NUMBER=r'(\d+(?:\.\d+)?)\s*(?:meters?\b|metres?\b|m\b|米)'
def quantities(text):
    s=text.lower()
    height_rules=[
        r'(?:altitude|height)(?:\s+of)?\s*'+NUMBER,
        r'(?:高度|定高)(?:为|在|约)?\s*'+NUMBER,
        NUMBER+r'\s*(?:的?高度|above (?:the )?(?:floor|ground))',
        r'\b(?:fly|cruise|hover|travel)\b[^.;]{0,60}?\bat\s+(?:about\s+)?'+NUMBER+r'(?!\s*(?:from|away|before|behind|in front))'
    ]
    length_rules=[
        r'(?:forward(?:\s+for)?|straight(?:\s+for)?|advance|cover|travel distance(?:\s+of)?|fly (?:for |a distance of ))\s*'+NUMBER,
        r'(?:向前(?:飞行)?|前进|航程|飞行距离|飞行长度)(?:为|约)?\s*'+NUMBER,
        NUMBER+r'\s*(?:forward\b|向前)'
    ]
    def metric(patterns):
        for pattern in patterns:
            m=re.search(pattern,s)
            if m:return float(m[1])
        return None
    return metric(height_rules),metric(length_rules)
