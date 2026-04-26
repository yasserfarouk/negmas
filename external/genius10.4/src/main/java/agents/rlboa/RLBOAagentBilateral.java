package agents.rlboa;

import genius.core.actions.Accept;
import genius.core.actions.EndNegotiation;
import genius.core.boaframework.BOAagentBilateral;
import genius.core.events.MultipartyNegoActionEvent;
import genius.core.events.NegotiationEvent;
import genius.core.events.SessionEndedNormallyEvent;

/**
 * This class orchestrates the learning process of any BOAAgentBilateral that
 * implements reinforcement learning into one of its components. To implement
 * such an agent one needs to extends this class and implement ones own versions
 * of the methods defined in the RLBOA interface.
 */

public abstract class RLBOAagentBilateral extends BOAagentBilateral implements RLBOA {

    @Override
    public abstract void agentSetup();

    @Override
    public void notifyChange(NegotiationEvent data) {

        // the event was an action (offer/accept/endnegotiation..)
        if (data instanceof MultipartyNegoActionEvent) {

            // Get relevant information from negotiation event
            MultipartyNegoActionEvent negoEvent = (MultipartyNegoActionEvent) data;

            // This will get handled by the SessionEndedNormallyEvent case
            if (negoEvent.getAction().getClass() == Accept.class || negoEvent.getAction().getClass() == EndNegotiation.class) {
                return;
            }

            // Observe state
            AbstractState newState = this.getStateRepresentation(negoEvent);

            double reward = this.getReward(negoEvent.getAgreement());
            boolean myTurn = negoEvent.getAction().getAgent() == this.getAgentID();

            if (newState.isTerminalState() || !myTurn) {

                this.observeEnvironment(reward, newState);

            }
        }
        // the negotation ended
        if (data instanceof SessionEndedNormallyEvent) {
            double reward = this.getReward(((SessionEndedNormallyEvent) data).getAgreement());
            this.observeEnvironment(reward, State.TERMINAL);
        }
    }
}
