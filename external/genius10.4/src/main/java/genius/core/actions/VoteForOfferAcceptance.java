package genius.core.actions;

import javax.xml.bind.annotation.XmlElement;
import javax.xml.bind.annotation.XmlRootElement;

import genius.core.AgentID;
import genius.core.Vote;

/**
 * This action can be used to indicate accept or reject of an offer. immutable
 * 
 * @author Reyhan
 */

@XmlRootElement
public class VoteForOfferAcceptance extends DefaultAction {

	@XmlElement
	protected Vote vote;

	public VoteForOfferAcceptance(AgentID party, Vote vote) {
		super(party);
		this.vote = vote;
	}

	public Vote getVote() {
		return vote;
	}

	public String toString() {
		return "Vote: " + (vote == null ? "null" : (vote == Vote.ACCEPT ? "Accept" : "Reject"));
	}

}
