package genius.core.exceptions;

import genius.core.parties.NegotiationParty;

/**
 * Exception illustrating that calculating a feature of the bidspace (for
 * example the Nash point) went wrong.
 */
public class NegotiationPartyTimeoutException extends Exception {
	protected NegotiationParty instigator;

	public NegotiationParty getInstigator() {
		return instigator;
	}

	public NegotiationPartyTimeoutException(NegotiationParty instigator) {
		super();
		this.instigator = instigator;
	}

	public NegotiationPartyTimeoutException(NegotiationParty instigator,
			String message) {
		super(message);
		this.instigator = instigator;
	}

	public NegotiationPartyTimeoutException(NegotiationParty instigator,
			String message, Throwable cause) {
		super(message, cause);
		this.instigator = instigator;
	}

	public NegotiationPartyTimeoutException(NegotiationParty instigator,
			Throwable cause) {
		super(cause);
		this.instigator = instigator;
	}
}