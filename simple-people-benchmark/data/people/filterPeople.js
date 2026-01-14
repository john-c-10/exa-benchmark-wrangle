const fs = require('fs');
const path = require('path');

const filePath = path.join(__dirname, 'simple_people_search.jsonl');
const outputPath = path.join(__dirname, 'simple_people_search_usa.jsonl');

let peopleData = [];

try {
    const data = fs.readFileSync(filePath, 'utf-8');
    // simple_people_search.jsonl is a JSON Lines file, so parse each line separately
    const lines = data.split(/\r?\n/).filter(Boolean);
    peopleData = lines.map(line => JSON.parse(line))
                      .filter(line => ((line.metadata.geo_name === 'United States') || line.metadata.geo_type === 'state' || line.metadata.geo_type === 'city'));
    // console.log(peopleData.slice(0, 5));
    // return;
    const writeStream = fs.createWriteStream(outputPath, { flags: 'w' });
    peopleData.forEach((person) => {
        writeStream.write(JSON.stringify(person) + '\n');
    });
    writeStream.end();
} catch (err) {
    console.error('Error reading or parsing JSONL file:', err);
    peopleData = [];
}
