package genius.core.protocol;

import java.util.ArrayList;
import java.util.List;

import genius.core.parties.Mediator;
import genius.core.parties.NegotiationParty;

/**
 * Base class for all mediator-based protocols
 *
 * @author David Festen
 */
public abstract class MediatorProtocol extends DefaultMultilateralProtocol {

	/**
	 * Returns the first mediator from a list of parties
	 *
	 * @param parties
	 *            The list of parties to find the mediator in
	 * @return The first mediator in the given list, or null if no such entity
	 *         exists.
	 */
	public static Mediator getMediator(List<NegotiationParty> parties) {

		// Mediator should be first item, but this should find it wherever it
		// is.
		for (NegotiationParty party : parties) {
			if (party instanceof Mediator) {
				return (Mediator) party;
			}
		}

		// default case, no mediator found, return null
		return null;
	}

	/**
	 * Get the non-mediator parties. assumption: there are only zero or one
	 * mediators in the list
	 *
	 * @param parties
	 *            The list of parties with mediator
	 * @return The given without the mediators
	 */
	public static List<NegotiationParty> getNonMediators(List<NegotiationParty> parties) {

		List<NegotiationParty> nonMediators = new ArrayList<NegotiationParty>(parties);
		nonMediators.remove(getMediator(parties));
		return nonMediators;
	}

}
