package genius.core.boaframework;

/**
 * Possible actions of an acceptance strategy.
 * 
 * @author Mark Hendrikx
 */
public enum Actions {
	/** Accept the opponent's offer. */
    Accept,
    /** Reject the opponent's offer. */
    Reject,
    /** Walk away from the negotiation. */
    Break;
}