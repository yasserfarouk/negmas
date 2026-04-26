package genius.core;

import java.util.ArrayList;

import genius.core.actions.Action;

/**
 *   @author Reyhan
 *
 */

public class NegoTurn {

	private int partyIndex;
	private ArrayList<Class> validActions;

	public NegoTurn(int partyIndex) {
		this.setPartyIndex(partyIndex);
		this.setValidActions(new ArrayList<Class>());
	}
	
	public NegoTurn(int partyIndex, ArrayList<Class> validNegoActions) {
		this.setPartyIndex(partyIndex);
		this.setValidActions(validNegoActions);
	}

	public NegoTurn(int partyIndex, Class validNegoAction) { //if there is only one valid actions use this constr.
		this.setPartyIndex(partyIndex);
		this.setValidActions(new ArrayList<Class>());
		this.addValidAction(validNegoAction);
		
	}
	public int getPartyIndex() {
		return partyIndex;
	}

	public void setPartyIndex(int partyIndex) {
		this.partyIndex = partyIndex;
	}

	public ArrayList<Class> getValidActions() {
		return validActions;
	}

	public void setValidActions(ArrayList<Class> validNegoActions) {
		this.validActions = validNegoActions;
	}	
	
	public void addValidAction(Class validNegoAction) {
		this.validActions.add(validNegoAction);
	}
	
	public void removeValidAction(Class validNegoAction) {
		this.validActions.remove(validNegoAction);
	}
	
	public void clearValidActions()	{
		this.setValidActions(new ArrayList<Class>());
	}
	
}
