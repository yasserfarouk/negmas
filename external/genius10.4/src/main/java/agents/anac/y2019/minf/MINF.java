package agents.anac.y2019.minf;

import java.io.PrintWriter;
import java.util.*;

import agents.anac.y2019.minf.etc.*;
import genius.core.AgentID;
import genius.core.Bid;
import genius.core.Domain;
import genius.core.actions.Accept;
import genius.core.actions.Action;
import genius.core.actions.EndNegotiation;
import genius.core.actions.Offer;
import genius.core.bidding.BidDetails;
import genius.core.boaframework.*;
import genius.core.uncertainty.AdditiveUtilitySpaceFactory;
import genius.core.uncertainty.BidRanking;
import genius.core.utility.AbstractUtilitySpace;

@SuppressWarnings("serial")
public class MINF extends BoaParty
{
	private Bid oppBid;
	private Action lastReceivedAction;
	private NegotiationInfo ni;

	@Override
	public void init(genius.core.parties.NegotiationInfo info)
	{
		ni = new NegotiationInfo(info.getUtilitySpace().getDiscountFactor(), info.getUtilitySpace().getReservationValue());
		AcceptanceStrategy 	ac  = new AC_Next(ni);
		OfferingStrategy 	os  = new TimeDependent_Offering(ni);
		OpponentModel 		om  = new HardHeadedFrequencyModel();
		OMStrategy			oms = new BestBid();
		lastReceivedAction 		= null;
		
		// All component parameters can be set below.
		Map<String, Double> noparams = Collections.emptyMap();
		Map<String, Double> osParams = new HashMap<String, Double>();
		osParams.put("CT", 0.998);

		// Initialize all the components of this party to the choices defined above
		configure(ac, noparams, 
				os,	osParams, 
				om, noparams,
				oms, noparams);
		super.init(info);
	}

	@Override
	public void receiveMessage(AgentID sender, Action opponentAction) {
		lastReceivedAction = opponentAction;
		// 1. if the opponent made a bid
		if (opponentAction instanceof Offer) {
			oppBid = ((Offer) opponentAction).getBid();

			// 2. store the opponent's trace
			try {
				BidDetails opponentBid = new BidDetails(oppBid,
						negotiationSession.getUtilitySpace().getUtility(oppBid),
						negotiationSession.getTime());
				negotiationSession.getOpponentBidHistory().add(opponentBid);
				ni.updateInfo(opponentBid.getMyUndiscountedUtil());
			} catch (Exception e) {
				e.printStackTrace();
			}
			// 3. if there is an opponent model, update it using the opponent's
			// bid
			if (opponentModel != null && !(opponentModel instanceof NoModel)) {
				if (omStrategy.canUpdateOM()) {
					opponentModel.updateModel(oppBid);
					ni.updateOwnInfo(opponentModel.getBidEvaluation(oppBid));
				} else {
					if (!opponentModel.isCleared()) {
						opponentModel.cleanUp();
					}
				}
			}
		}
	}

	@Override
	public Action chooseAction(List<Class<? extends Action>> possibleActions) {
		BidDetails bid;

		// if our history is empty, then make an opening bid
		if (negotiationSession.getOwnBidHistory().getHistory().isEmpty()) {
			bid = offeringStrategy.determineOpeningBid();
			if (!(lastReceivedAction instanceof Offer)) { ni.setFirst(true); }
			//outputBid();
		} else {
			// else make a normal bid
			bid = offeringStrategy.determineNextBid();
			if (offeringStrategy.isEndNegotiation()) {
				return new EndNegotiation(getPartyId());
			}
		}

		// if the offering strategy made a mistake and didn't set a bid: accept
		if (bid == null) {
			System.out.println("Error in code, null bid was given");
			return new Accept(getPartyId(), oppBid);
		} else {
			ni.updateMyInfo(bid.getMyUndiscountedUtil());
			offeringStrategy.setNextBid(bid);
		}

		// check if the opponent bid should be accepted
		Actions decision = Actions.Reject;
		if (!negotiationSession.getOpponentBidHistory().getHistory()
				.isEmpty()) {
			decision = acceptConditions.determineAcceptability();
		}

		// check if the agent decided to break off the negotiation
		if (decision.equals(Actions.Break)) {
			System.out.println("send EndNegotiation");
			return new EndNegotiation(getPartyId());
		}
		// if agent does not accept, it offers the counter bid
		if (decision.equals(Actions.Reject)) {
			negotiationSession.getOwnBidHistory().add(bid);
			return new Offer(getPartyId(), bid.getBid());
		} else {
			return new Accept(getPartyId(), oppBid);
		}
	}

	@Override
	public AbstractUtilitySpace estimateUtilitySpace()
	{
		Domain domain = getDomain();
		BidRanking bidRanking = getUserModel().getBidRanking();
		LP_Estimation lpe = new LP_Estimation(domain, bidRanking);
		AdditiveUtilitySpaceFactory additiveUtilitySpaceFactory;

		try {
			additiveUtilitySpaceFactory = lpe.Estimation();
		} catch (Exception e){
			e.printStackTrace();
			additiveUtilitySpaceFactory = new AdditiveUtilitySpaceFactory(domain);
			additiveUtilitySpaceFactory.estimateUsingBidRanks(bidRanking);
		}

        /*for (IssueDiscrete i : issues) {
            double weight = additiveUtilitySpaceFactory.getUtilitySpace().getWeight(i.getNumber());
            System.out.println("W:"+weight);
            for (ValueDiscrete v : i.getValues()){
                System.out.println("V:"+additiveUtilitySpaceFactory.getUtility(i, v));
            }
        }*/

		return additiveUtilitySpaceFactory.getUtilitySpace();
	}
	
	@Override
	public String getDescription() 
	{
		return "ANAC 2019";
	}

    public void outputBid(){
		try{
			List<BidDetails> allOutcomes = negotiationSession.getOutcomeSpace().getAllOutcomes();
			PrintWriter pw = new PrintWriter("result.csv");
			pw.println("sep=;");

			for (BidDetails bd : allOutcomes){
				pw.println(bd.getBid().toString() + ";" + bd.getMyUndiscountedUtil());
			}
			pw.close();
		} catch (Exception e){
			e.printStackTrace();
		}
    }
}
