"""Task-order counterfactual independent of any route files. General object order for the three detector labels."""
import json
import re
from pathlib import Path
from policy import Policy

NOUNS = {'air conditioner':'ac','laptop':'laptop','notebook computer':'laptop','sofa':'sofa','couch':'sofa'}
NP = re.compile(r'\b('+'|'.join(re.escape(n) for n in NOUNS)+r')\b')
# Relative-location phrases introduce a reference landmark, not a second inspection target.
RELATION = re.compile(r'\b(?:above|below|under|beside|behind|in front of|next to)\b')
def parse_targets(instruction):
    result=[]
    for clause in re.split(r'[.,;]|\bthen\b|\band (?=(?:turn|inspect|search|approach|find|face|observe)\b)',instruction.lower()):
        action=RELATION.split(clause,maxsplit=1)[0]
        for w in NP.findall(action):
            obj=NOUNS[w]
            if not result or obj!=result[-1]:result.append(obj)
    return result
class TextOrderPolicy(Policy):
    def __init__(self,instruction):
        super().__init__(instruction)
        self.objects=parse_targets(instruction)
        self.search=bool(self.objects and re.search(r'\b(?:find|search|locate)\b',instruction.lower()))

if __name__=='__main__':
    from score import ROOT,OUT,get,save
    texts=[get(p)['instruction'] for p in sorted((ROOT.parent/'runs/gimbal/model_fixed').glob('*/summary.json'))]
    eq=[]
    for text in texts:
        old=Policy(text);new=TextOrderPolicy(text)
        # All remaining methods are inherited verbatim. Equality of initial fields implies equal outputs for equal inputs.
        for attr in ['objects','search','climb','alt','profile','index','confirm','scan','last_yaw','desc','z0','frame']:
            assert getattr(old,attr)==getattr(new,attr),(text,attr)
        eq.append(dict(text=text,targets=new.objects))
    examples=[]
    for text,seq in [('Find the black sofa, then turn toward the open laptop.',['sofa','laptop']),
                     ('Find the open laptop, then turn toward the black sofa.',['laptop','sofa']),
                     ('Find the couch, then approach the air conditioner.',['sofa','ac']),
                     ('Approach the air conditioner above the sofa.',['ac'])]:
        orig=Policy(text).objects;generic=TextOrderPolicy(text).objects;assert generic==seq
        examples.append(dict(text=text,expected=seq,frozen=orig,parsed=generic))
    save(OUT/'language_control.json',dict(observed_instructions_equivalent=eq,unflown_language_counterfactuals=examples))
