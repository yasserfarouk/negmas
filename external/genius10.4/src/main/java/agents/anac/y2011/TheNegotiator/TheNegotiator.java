package agents.anac.y2011.TheNegotiator;

import genius.core.Agent;
import genius.core.SupportedNegotiationSetting;
import genius.core.actions.Action;

/**
 * The TheNegotiator class specifies a negotiation agent in the GENIUS domain.
 * 
 * @author Alex Dirkzwager, Mark Hendrikx, Julian de Ruiter
 */
public class TheNegotiator extends Agent {

	// reference to the decider (governs which action should be performed each
	// turn)
	private Decider decider;

	/**
	 * init is called when a next session starts with the same partner.
	 */
	@Override
	public void init() {
		decider = new Decider(this);
	}

	@Override
	public String getVersion() {
		return "3.0";
	}

	@Override
	public String getName() {
		return "The Negotiator";
	}

	/**
	 * Store the actions made by a partner.
	 * 
	 * @param action
	 *            by partner in current turn
	 */
	@Override
	public void ReceiveMessage(Action partnerAction) {
		decider.setPartnerMove(partnerAction);
	}

	/**
	 * Choose an action to perform.
	 */
	@Override
	public Action chooseAction() {
		return decider.makeDecision();
	}

	@Override
	public SupportedNegotiationSetting getSupportedNegotiationSetting() {
		return SupportedNegotiationSetting.getLinearUtilitySpaceInstance();
	}

	@Override
	public String getDescription() {
		return "ANAC2011";
	}
}