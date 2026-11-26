"""Prospective supplementary starts for rear bearings; same frozen policy, executor and detector."""
import sys
import run
OFFSETS={'room_inspect_ac':(.36,.12,165), 'corridor_high_return':(.45,-.10,-160)}
if __name__=='__main__':
    run.OFFSETS=OFFSETS
    run.OUT=run.OUT/'reverse'
    run.OUT.mkdir(exist_ok=True)
    detector=None
    for name in OFFSETS:
        for arm in ['baseline','hint','guard','both']:
            if name.startswith('room') and arm in ('hint','both') and detector is None:
                from detector import Detector
                detector=Detector()
            run.run(name,True,arm,sys.argv[1],detector)
