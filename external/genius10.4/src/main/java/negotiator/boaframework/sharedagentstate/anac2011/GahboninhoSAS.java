package negotiator.boaframework.sharedagentstate.anac2011;

import genius.core.boaframework.NegotiationSession;
import genius.core.boaframework.SharedAgentState;
import genius.core.timeline.TimeLineInfo;
import genius.core.utility.AdditiveUtilitySpace;
import negotiator.boaframework.sharedagentstate.anac2011.gahboninho.GahboninhoOM;
import negotiator.boaframework.sharedagentstate.anac2011.gahboninho.IssueManager;

/**
 * This is the shared code of the acceptance condition and bidding strategy of
 * ANAC 2011 Gahboninho. The code was taken from the ANAC2011 Gahboninho and
 * adapted to work within the BOA framework.
 * 
 * @author Mark Hendrikx
 */
public class GahboninhoSAS extends SharedAgentState {
	private GahboninhoOM om;
	private IssueManager im;
	private AdditiveUtilitySpace utilSpace;
	private TimeLineInfo timeline;
	private int firstActions = 40;
	private NegotiationSession negotiationSession;

	public GahboninhoSAS(NegotiationSession negoSession) {
		this.negotiationSession = negoSession;
		this.utilSpace = (AdditiveUtilitySpace) negoSession.getUtilitySpace();
		this.timeline = negoSession.getTimeline();
		initObjects();
		NAME = "Gahboninho";
	}

	public void initObjects() {
		om = new GahboninhoOM(utilSpace, timeline);
		im = new IssueManager(negotiationSession, timeline, om);
		im.setNoise(im.getNoise() * im.GetDiscountFactor());
	}

	public GahboninhoOM getOpponentModel() {
		return om;
	}

	public IssueManager getIssueManager() {
		return im;
	}

	public int getFirstActions() {
		return firstActions;
	}

	public void decrementFirstActions() {
		--firstActions;
	}
}