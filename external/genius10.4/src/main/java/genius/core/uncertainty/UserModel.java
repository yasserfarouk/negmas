package genius.core.uncertainty;

import genius.core.Domain;

/**
 * @author Tim Baarslag, Dimitrios Tsimpoukis
 * 
 * This class specifies uncertainty to the agents in the form of a user model.
 */
public class UserModel
{
	protected BidRanking bidRanking;
	
	/**
	 * @param allOutcomes
	 */
	public UserModel(BidRanking bidRanking) {
		this.bidRanking = bidRanking;
	}
	
	@Override
	public String toString() {
		return bidRanking.toString();
	}


	public BidRanking getBidRanking() {
		return bidRanking;
	}

	public Domain getDomain() {
		return this.bidRanking.getBidOrder().get(0).getDomain();
	}
	
}


