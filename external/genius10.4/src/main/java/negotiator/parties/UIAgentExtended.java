package negotiator.parties;

import java.awt.Toolkit;
import java.util.ArrayList;

import javax.swing.JOptionPane;

import genius.core.Agent;
import genius.core.Bid;
import genius.core.SupportedNegotiationSetting;
import genius.core.actions.Accept;
import genius.core.actions.Action;
import genius.core.actions.EndNegotiation;
import genius.core.actions.Offer;
import genius.core.utility.AdditiveUtilitySpace;

/**
 * 
 * @author W.Pasman, modified version of Dmytro's UIAgent
 */
public class UIAgentExtended extends Agent {
	private Action opponentAction = null;
	private Bid myPreviousBid = null;

	// Alina added...
	private Bid oppPreviousBid = null;
	protected int bidCounter = 0;
	protected NegoRoundData roundData;
	protected ArrayList<NegoRoundData> historyOfBids = null;

	/** Creates a new instance of UIAgent */

	/**
	 * One agent will be kept alive over multiple sessions. Init will be called
	 * at the start of each nego session.
	 */
	@Override
	public String getVersion() {
		return "2.0";
	}

	public void init() {

		System.out.println("try to init UIAgent");
		System.out.println("Utility Space initialized: " + utilitySpace);
		historyOfBids = new ArrayList<NegoRoundData>();

	}

	private EnterBidDialogExtended getDialog() throws Exception {
		EnterBidDialogExtended ui = new EnterBidDialogExtended(this, null,
				true, (AdditiveUtilitySpace) utilitySpace, oppPreviousBid);
		// alina: dialog in the center- doesnt really work
		Toolkit t = Toolkit.getDefaultToolkit();
		int x = (int) ((t.getScreenSize().getWidth() - ui.getWidth()) / 2);
		int y = (int) ((t.getScreenSize().getHeight() - ui.getHeight()) / 2);
		ui.setLocation(x, y);
		return ui;
	}

	public void ReceiveMessage(Action opponentAction) {
		this.opponentAction = opponentAction;
		if (opponentAction instanceof Accept)
			JOptionPane.showMessageDialog(null,
					"Opponent accepted your last offer.");

		if (opponentAction instanceof EndNegotiation)
			JOptionPane.showMessageDialog(null,
					"Opponent canceled the negotiation session");

		return;
	}

	public Action chooseAction() {
		try {
			return chooseAction1();
		} catch (Exception e) {
			e.printStackTrace();
		}
		return null;
	}

	public Action chooseAction1() throws Exception {
		Action action = getDialog().askUserForAction(opponentAction,
				myPreviousBid);
		if ((action != null) && (action instanceof Offer)) {
			myPreviousBid = ((Offer) action).getBid();
			if (opponentAction != null) {
				oppPreviousBid = ((Offer) opponentAction).getBid();
				roundData = new NegoRoundData(oppPreviousBid, myPreviousBid);
				historyOfBids.add(roundData);
			}
			// does this happen only the first time?
			else {
				roundData = new NegoRoundData(null, myPreviousBid);
				historyOfBids.add(roundData);
			}
			bidCounter++;
		}

		return action;
	}

	public boolean isUIAgent() {
		return true;
	}

	public Bid getMyPreviousBid() {
		return myPreviousBid;
	}

	public Bid getOppPreviousBid() {
		return oppPreviousBid;
	}

	@Override
	public SupportedNegotiationSetting getSupportedNegotiationSetting() {
		return SupportedNegotiationSetting.getLinearUtilitySpaceInstance();
	}
}

class NegoRoundData {
	private Bid lastOppBid;
	private Bid ourLastBid;

	public NegoRoundData(Bid lastOppBid, Bid ourLastBid) {
		this.lastOppBid = lastOppBid;
		this.ourLastBid = ourLastBid;
	}

	public Bid getOppentBid() {
		return lastOppBid;
	}

	public Bid getOurBid() {
		return ourLastBid;
	}

}
