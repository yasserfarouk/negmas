package negotiator.boaframework.opponentmodel;

import javax.swing.JOptionPane;

import genius.core.Bid;
import genius.core.boaframework.OpponentModel;
import genius.core.protocol.BilateralAtomicNegotiationSession;
import genius.core.tournament.TournamentConfiguration;
import genius.core.utility.AdditiveUtilitySpace;

/**
 * An opponent model symbolizing perfect knowledge about the opponent's
 * preferences. Note that for using this model experimentalSetup should be
 * enabled in global. Since this extends OpponentModel, it only supports
 * {@link AdditiveUtilitySpace}.
 * 
 * Tim Baarslag, Koen Hindriks, Mark Hendrikx, Alex Dirkzwager and Catholijn M.
 * Jonker. Decoupling Negotiating Agents to Explore the Space of Negotiation
 * Strategies
 * 
 * @author Mark Hendrikx
 */
public class PerfectModel extends OpponentModel {

	@Override
	public void setOpponentUtilitySpace(
			BilateralAtomicNegotiationSession session) {

		if (TournamentConfiguration.getBooleanOption(
				"accessPartnerPreferences", false)) {
			opponentUtilitySpace = (AdditiveUtilitySpace) session
					.getAgentAUtilitySpace();
			if (negotiationSession.getUtilitySpace().getFileName()
					.equals(opponentUtilitySpace.getFileName())) {
				opponentUtilitySpace = (AdditiveUtilitySpace) session
						.getAgentBUtilitySpace();
			}
		} else {
			JOptionPane
					.showMessageDialog(
							null,
							"This opponent model needs access to the opponent's\npreferences. See tournament options.",
							"Model error", 0);
			System.err.println("Global.experimentalSetup should be enabled!");
		}
	}

	@Override
	public void setOpponentUtilitySpace(
			AdditiveUtilitySpace opponentUtilitySpace) {
		this.opponentUtilitySpace = opponentUtilitySpace;
	}

	@Override
	public double getBidEvaluation(Bid bid) {
		try {
			return opponentUtilitySpace.getUtility(bid);
		} catch (Exception e) {
			e.printStackTrace();
		}
		return 0;
	}

	@Override
	public String getName() {
		return "Perfect Model";
	}

	public void updateModel(Bid opponentBid, double time) {
	}
}