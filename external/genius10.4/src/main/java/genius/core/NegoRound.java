package genius.core;

import java.util.ArrayList;

import genius.core.actions.Action;

/**
 *   @author Reyhan
 *
 */

public class NegoRound {
	
	private ArrayList<NegoTurn> partyActionList;
	private int currentTurnIndex;
	private int currentRoundNo;
	private ArrayList<Class> actionsTerminateSessionWithSuccess;
	private ArrayList<Class> actionsTerminateSessionWithFailure;
	
	public NegoRound() {
		
		this.setPartyActionList(new ArrayList<NegoTurn>());
		this.setActionsTerminateSessionWithFailure(new ArrayList<Class>());
		this.setActionsTerminateSessionWithSuccess(new ArrayList<Class>());
		this.currentTurnIndex=0;
		this.setCurrentRoundNo(1);	
		
	}
	
	public NegoRound(ArrayList<NegoTurn> partyActionList, int numberOfTurnInARound) {
		
		this.setPartyActionList(partyActionList);
		this.setActionsTerminateSessionWithFailure(new ArrayList<Class>());
		this.setActionsTerminateSessionWithSuccess(new ArrayList<Class>());
		this.currentTurnIndex=0;
		this.setCurrentRoundNo(1);
	}

	public NegoRound(ArrayList<NegoTurn> partyActionList, ArrayList<Class> actionsTerminateWithSuccess, ArrayList<Class> actionsTerminateWithFailure,int numberOfTurnInARound) {
		
		this.setPartyActionList(partyActionList);
		this.setActionsTerminateSessionWithFailure(actionsTerminateWithFailure);
		this.setActionsTerminateSessionWithSuccess(actionsTerminateWithSuccess);
		this.currentTurnIndex=0;
		this.setCurrentRoundNo(1);
	}
	
	//clone
	public NegoRound(NegoRound negoRound){
		
		this.partyActionList=negoRound.getPartyActionList();
		this.currentTurnIndex=negoRound.getCurrentTurnIndex();
		this.currentRoundNo=negoRound.getCurrentRoundNo();
		this.actionsTerminateSessionWithFailure=negoRound.getActionsTerminateSessionWithFailure();
		this.actionsTerminateSessionWithSuccess=negoRound.getActionsTerminateSessionWithSuccess();
	}

	protected int getCurrentTurnIndex(){
		return currentTurnIndex;
	}
	
	public ArrayList<Class> getActionsTerminateSessionWithFailure(){
		return actionsTerminateSessionWithFailure;
	}
	
	public void setActionsTerminateSessionWithFailure(ArrayList<Class> actions) {
		actionsTerminateSessionWithFailure=actions;
	}
	
	public void addActionTerminateSessionWithFailure(Class action) {
		actionsTerminateSessionWithFailure.add(action);
	}
	
	public void removeActionTeminateSessionWithFailure (Class action) {
		
		actionsTerminateSessionWithFailure.remove(action);
	}
	
	public void clearActionsTerminateSessionWithFailure () {
		actionsTerminateSessionWithFailure=new ArrayList<Class>();
	}

	public ArrayList<Class> getActionsTerminateSessionWithSuccess(){
		return actionsTerminateSessionWithSuccess;
	}
	
	public void setActionsTerminateSessionWithSuccess(ArrayList<Class> actions) {
		actionsTerminateSessionWithSuccess=actions;
	}
	
	public void addActionTerminateSessionWithSuccess(Class action) {
		actionsTerminateSessionWithSuccess.add(action);
	}
	
	public void removeActionTeminateSessionWithSuccess (Class action) {
		
		actionsTerminateSessionWithSuccess.remove(action);
	}
	
	public void clearActionsTerminateSessionWithSuccess () {
		actionsTerminateSessionWithSuccess=new ArrayList<Class>();
	}
	
	public int getCurrentRoundNo() {
		return currentRoundNo;
	}

	public void setCurrentRoundNo(int currentRoundNo) {
		this.currentRoundNo = currentRoundNo;
	}

	public ArrayList<NegoTurn> getPartyActionList() {
		return partyActionList;
	}

	public void setPartyActionList(ArrayList<NegoTurn> partyActionList) {
		this.partyActionList = partyActionList;
	}
	
	
	public void addPartyActions(NegoTurn partyAction) {
		partyActionList.add(partyAction);
	}
	
	public NegoTurn getCurrentPartyAndValidActions() {
		return (partyActionList.get(currentTurnIndex));
	}

	public int getCurrentPartyIndex() 	{
		return (partyActionList.get(currentTurnIndex).getPartyIndex());
	}
	
	public ArrayList<Class> getCurrentPartysValidActions()	{
		return (partyActionList.get(currentTurnIndex).getValidActions());
	}
	
	public boolean setNextTurn(){
		
	   currentTurnIndex= (currentTurnIndex+1) % partyActionList.size();
	   if (currentTurnIndex==0){		   
		   currentRoundNo++;
		   return true;
	   }
	   return false;
	}
	
    
	public boolean isCurrentActionValid(Action currentAction){
		
		if (getCurrentPartysValidActions().contains(currentAction.getClass()))
			return true;
		else			
			return false;	
	}

	public boolean isDeadlineReached(int maxRound){
		
		if (currentRoundNo>maxRound)
			return true;
		else return false;
	}
	
	public boolean doesTerminateWithSuccess(Action currentAction) {
		
		if (actionsTerminateSessionWithSuccess.contains(currentAction.getClass()))
			return true;
		else 
			return false;			
	}
	
	public boolean doesTerminateWithFailure(Action currentAction) {
		
		if (actionsTerminateSessionWithFailure.contains(currentAction.getClass()))
			return true;
		else 
			return false;			
	}
	
}

