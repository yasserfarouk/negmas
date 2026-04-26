package negotiator.repository;

import java.io.StringWriter;

import javax.xml.bind.JAXBContext;
import javax.xml.bind.JAXBException;
import javax.xml.bind.Marshaller;

import org.junit.Test;

import genius.core.repository.BoaPartiesList;
import genius.core.repository.BoaPartyRepository;

/**
 * 
 * FIXME we should make unit test for {@link BoaPartyRepository} but I can't
 * figure out how to do it. #1400
 */
public class BoaPartyRepoTest {

	@Test
	public void testNewEmptyList() throws JAXBException {
		BoaPartiesList list = new BoaPartiesList();
		StringWriter writer = new StringWriter();

		JAXBContext context = JAXBContext.newInstance(BoaPartiesList.class);
		Marshaller m = context.createMarshaller();
		m.marshal(list, writer);
		System.out.println(writer.toString());

	}
}
