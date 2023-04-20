var vt2geojson = require('@mapbox/vt2geojson');
const FileSystem = require("fs");

// remote file
// vt2geojson({
//     uri: 'http://api.example.com/9/150/194.mvt',
//     layer: 'layer_name'
// }, function (err, result) {
//     if (err) throw err;
//     console.log(result); // => GeoJSON FeatureCollection
// });

// local file

function showResults(err, result){
    if (err) throw err;
    // console.log(result); // => GeoJSON FeatureCollection

    console.log(typeof result)
    console.log(JSON.stringify(result['features'][1]))

    FileSystem.writeFile('35068_24354.json', JSON.stringify(result), (error) => {
        if (error) throw error;
      });
}


vt2geojson({
    uri: './35068_24354.mvt',
    // layer: 'layer_name',
    z: 16,
    x: 35068,
    y: 24354
}, showResults);

