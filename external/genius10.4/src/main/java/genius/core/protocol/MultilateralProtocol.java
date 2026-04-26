package genius.core.protocol;

import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutionException;

import genius.core.AgentID;
import genius.core.Bid;
import genius.core.actions.Action;
import genius.core.exceptions.NegotiationPartyTimeoutException;
import genius.core.parties.NegotiationParty;
import genius.core.session.ActionException;
import genius.core.session.Round;
import genius.core.session.Session;
import genius.core.session.SessionManager;
import genius.core.session.Turn;

/**
 * The protocol describes if the negotiation is finished, what the agreement is,
 * which actions can be done in the next round.
 * <p>
 * The protocol is executed for example by the
 * {@link genius.core.session.SessionManager}.
 * 
 * <h1>Specification</h1> A protocol should be used as follows.
 * <ol>
 * <li>The first time, {@link #beforeSession(Session, List)} should be called
 * and all agents should receive the actions accordingly.
 * <li>For each round in the session:
 * <ol>
 * <li>{@link #getRoundStructure(List, Session)} should be called to determine
 * the {@link Round}
 * <li>For each {@link Turn} in the {@link Round} :
 * <ol>
 * <li>the {@link Turn#getParty()} agent should be called with the specified
 * allowed actions.
 * <li>The agent returns a picked action
 * <li>{@link #getActionListeners(List)} should be called to get a list of which
 * agents need to hear the picked action of this agent.
 * </ol>
 * <li>{@link #isFinished(Session, List)} should be checked after the round is
 * complete to see if there are other rounds . If so, repeat
 * </ol>
 * <li>When a session is completely done, {@link #afterSession(Session, List)}
 * should be called
 * </ol>
 * 
 * @author David Festen
 */
public interface MultilateralProtocol {

	/**
	 * Get the structure of the current round. Each round, this method receives
	 * a list of all the {@link genius.core.parties.NegotiationParty} and the
	 * complete {@link Session} which can be used to diversify the round
	 * structure at some point during the session.
	 *
	 * @param parties
	 *            The parties currently participating
	 * @param session
	 *            The complete session history
	 * @return A list of possible actions
	 */
	Round getRoundStructure(List<NegotiationParty> parties, Session session);

	/**
	 * Returns a list of Actions to be sent to
	 * {@link NegotiationParty#receiveMessage(AgentID, Action)} . This will get
	 * called just before the session starts. If some initialization with needs
	 * to be done by the protocol, it can be done here.
	 *
	 * 
	 * @param session
	 *            The upcoming {@link Session}
	 * @param parties
	 *            The {@link NegotiationParty}s that will participate in the
	 *            session
	 */
	Map<NegotiationParty, List<Action>> beforeSession(Session session, List<NegotiationParty> parties)
			throws NegotiationPartyTimeoutException, ExecutionException, InterruptedException;

	/**
	 * This will get called just after ending the session. If the protocol needs
	 * to do some post session steps, it can be done here. Protocols should not
	 * handle {@link NegotiationParty#negotiationEnded(Bid)} as these are always
	 * called by the {@link SessionManager}.
	 *
	 * @param session
	 *            The session instance that was used for the session
	 * @param parties
	 *            The parties that participated in the session
	 */
	void afterSession(Session session, List<NegotiationParty> parties);

	/**
	 * Apply the action according to the protocol. All actions done by all
	 * agents come through this method. The protocol should check here if the
	 * agent's action is actually allowed and contains the proper data.
	 *
	 * @param action
	 *            action to apply. The Agent ID in the action already has been
	 *            checked when this is called.
	 * @param session
	 *            the current state of this session
	 * @throws ActionException
	 *             if the proposed action is illegal according to the protocol.
	 */
	void applyAction(Action action, Session session) throws ActionException;

	/**
	 * Check if the protocol is done or still busy. If this method returns true,
	 * the {@link genius.core.session.SessionManager} will not start a new
	 * {@link Round} after the current one. It will however finish all the turns
	 * described in the
	 * {@link #getRoundStructure(java.util.List, genius.core.session.Session)}
	 * method.
	 *
	 * @param session
	 *            the current state of this session
	 * @param parties
	 *            all the parties involved in the negotiation
	 * @return true if the protocol is finished
	 */
	boolean isFinished(Session session, List<NegotiationParty> parties);

	/**
	 * Get a map of parties that are listening to each other's response. All
	 * these listeners should be informed whenever a party takes an action. See
	 * also the default implementations
	 * {@link DefaultMultilateralProtocol#listenToAll(List)} and
	 * {@link DefaultMultilateralProtocol#listenToNone(List)}
	 *
	 * @param parties
	 *            The parties involved in the current negotiation
	 * @return A map where the key is a {@link NegotiationParty} that is
	 *         responding to a {@link NegotiationParty#chooseAction(List)}
	 *         event, and the value is a list of {@link NegotiationParty}s that
	 *         are listening to that key party's response.
	 */
	Map<NegotiationParty, List<NegotiationParty>> getActionListeners(List<NegotiationParty> parties);

	/**
	 * This method should return the current agreement.
	 * <p/>
	 * Some protocols only have an agreement at the negotiation session, make
	 * sure that this method returns null until the end of the session in that
	 * case, because this method might be queried at intermediary steps.
	 *
	 * @param session
	 *            The complete session history up to this point
	 * @param parties
	 *            The parties involved in the current negotiation
	 * @return The agreed upon bid or null if no agreement
	 */
	Bid getCurrentAgreement(Session session, List<NegotiationParty> parties);

	/**
	 * Gets the number of parties that currently agree to the offer. For
	 * protocols that either have an agreement or not, you can set this number
	 * to 0 until an agreement is found, and then set this value to the number
	 * of parties.
	 *
	 * @param session
	 *            the current state of this session
	 * @param parties
	 *            The parties currently participating
	 * @return the number of parties agreeing to the current agreement
	 */
	int getNumberOfAgreeingParties(Session session, List<NegotiationParty> parties);

	/**
	 * Overwrites the rest of the protocol and sets the protocol state to finish
	 */
	void endNegotiation();

	/**
	 * Overwrites the rest of the protocol and sets the protocol state to finish
	 *
	 * @param reason
	 *            Optionally give a reason why the protocol is finished.
	 */
	void endNegotiation(String reason);

}
