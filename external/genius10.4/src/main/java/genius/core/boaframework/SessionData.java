package genius.core.boaframework;

import java.io.Serializable;

/**
 * In a BOAagent each component should be able to store data to be used in the next
 * negotiation session. Unfortunately, the ANAC2013 implementation only allows us to
 * store a single object. Therefore, this object packs the object of each of the three
 * BOA components (bidding strategy, etc.) together as a single object.
 * 
 * @author Mark Hendrikx
 */
public class SessionData implements Serializable {

	private static final long serialVersionUID = -2269008062554989046L;

	/** Object saved by the bidding strategy. */
	private Serializable biddingStrategyData;
	/** Object saved by the opponent model. */
	private Serializable opponentModelData;
	/** Object saved by the acceptance strategy. */
	private Serializable acceptanceStrategyData;
	/** Boolean used to mark that the object changed to avoid writing the same object to file. */
	private boolean changed;
	
	public SessionData() {
		biddingStrategyData = null;
		opponentModelData = null;
		acceptanceStrategyData = null;
		changed = false;
	}
	
	/**
	 * Returns the data stored by the given BOA component.
	 * @param type of the BOA component.
	 * @return component saved by this BOA component, null is no object saved.
	 */
	public Serializable getData(BoaType type) {
		Serializable result = null;
		if (type == BoaType.BIDDINGSTRATEGY) {
			result = biddingStrategyData;
		} else if (type == BoaType.OPPONENTMODEL) {
			result = opponentModelData;
		} else if (type == BoaType.ACCEPTANCESTRATEGY) {
			result = acceptanceStrategyData;
		}
		return result;
	}

	/**
	 * Method used to set the data to be saved by a BOA component.
	 * This method should not be called directly, use the storeData()
	 * and loadDate() of the component instead.
	 * 
	 * @param component from which the data is to be saved.
	 * @param data to be saved.
	 */
	public void setData(BoaType component, Serializable data) {
		if (component == BoaType.BIDDINGSTRATEGY) {
			biddingStrategyData = data;
		} else if (component == BoaType.OPPONENTMODEL) {
			opponentModelData = data;
		} else if (component == BoaType.ACCEPTANCESTRATEGY) {
			acceptanceStrategyData = data;
		}
		changed = true;
	}

	/**
	 * @return true if save was saved by a component.
	 */
	public boolean isEmpty() {
		return biddingStrategyData == null && opponentModelData == null && acceptanceStrategyData == null;
	}

	/**
	 * @return true if object was after its creation.
	 */
	public boolean isChanged() {
		return changed;
	}

	/**
	 * Sets that all changes have been processed.
	 */
	public void changesCommitted() {
		changed = false;
	}
}