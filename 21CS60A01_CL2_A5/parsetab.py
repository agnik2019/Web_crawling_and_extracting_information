
# parsetab.py
# This file is automatically generated. Do not edit.
# pylint: disable=W,C,R
_tabversion = '3.10'

_lr_method = 'LALR'

_lr_signature = 'CLOSEDIV CLOSESPAN EACTIVECASE LACTIVECASE LDEATH LRECOVERCASE LTOTALCASE NUM OPENSPANstart : total_cases\n                | recovered_cases \n                | deaths\n                | active_cases \n             total_cases : LTOTALCASE pnum CLOSESPANrecovered_cases : LRECOVERCASE OPENSPAN pnum CLOSESPAN CLOSEDIVdeaths : LDEATH OPENSPAN pnum CLOSESPAN CLOSEDIVactive_cases : LACTIVECASE pnum EACTIVECASEpnum : NUMpnum : NUM pnum'
    
_lr_action_items = {'LTOTALCASE':([0,],[6,]),'LRECOVERCASE':([0,],[7,]),'LDEATH':([0,],[8,]),'LACTIVECASE':([0,],[9,]),'$end':([1,2,3,4,5,15,19,22,23,],[0,-1,-2,-3,-4,-5,-8,-6,-7,]),'NUM':([6,9,11,12,13,],[11,11,11,11,11,]),'OPENSPAN':([7,8,],[12,13,]),'CLOSESPAN':([10,11,16,17,18,],[15,-9,-10,20,21,]),'EACTIVECASE':([11,14,16,],[-9,19,-10,]),'CLOSEDIV':([20,21,],[22,23,]),}

_lr_action = {}
for _k, _v in _lr_action_items.items():
   for _x,_y in zip(_v[0],_v[1]):
      if not _x in _lr_action:  _lr_action[_x] = {}
      _lr_action[_x][_k] = _y
del _lr_action_items

_lr_goto_items = {'start':([0,],[1,]),'total_cases':([0,],[2,]),'recovered_cases':([0,],[3,]),'deaths':([0,],[4,]),'active_cases':([0,],[5,]),'pnum':([6,9,11,12,13,],[10,14,16,17,18,]),}

_lr_goto = {}
for _k, _v in _lr_goto_items.items():
   for _x, _y in zip(_v[0], _v[1]):
       if not _x in _lr_goto: _lr_goto[_x] = {}
       _lr_goto[_x][_k] = _y
del _lr_goto_items
_lr_productions = [
  ("S' -> start","S'",1,None,None,None),
  ('start -> total_cases','start',1,'p_start','main_a5.py',367),
  ('start -> recovered_cases','start',1,'p_start','main_a5.py',368),
  ('start -> deaths','start',1,'p_start','main_a5.py',369),
  ('start -> active_cases','start',1,'p_start','main_a5.py',370),
  ('total_cases -> LTOTALCASE pnum CLOSESPAN','total_cases',3,'p_total_cases','main_a5.py',407),
  ('recovered_cases -> LRECOVERCASE OPENSPAN pnum CLOSESPAN CLOSEDIV','recovered_cases',5,'p_recovered_cases','main_a5.py',413),
  ('deaths -> LDEATH OPENSPAN pnum CLOSESPAN CLOSEDIV','deaths',5,'p_deaths','main_a5.py',419),
  ('active_cases -> LACTIVECASE pnum EACTIVECASE','active_cases',3,'p_active_cases','main_a5.py',425),
  ('pnum -> NUM','pnum',1,'p_pnum','main_a5.py',431),
  ('pnum -> NUM pnum','pnum',2,'p_pnum_multi','main_a5.py',436),
]
