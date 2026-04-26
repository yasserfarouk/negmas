package genius.core.parties;

import genius.core.protocol.SimpleMediatorBasedProtocol;

/**
 * Base class for all mediator parties.
 * 
 * A mediator is running on a protocol that supports mediators, such as the
 * {@link SimpleMediatorBasedProtocol}.
 *
 * @author David Festen, W.Pasman
 * 
 */
public interface Mediator extends NegotiationParty {

}
